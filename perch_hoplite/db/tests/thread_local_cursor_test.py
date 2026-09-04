# coding=utf-8
# Copyright 2026 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for thread-local cursor handling in SQLiteUSearchDB."""

import shutil
import tempfile
import threading
import time

import numpy as np
from perch_hoplite.db.tests import test_utils

from absl.testing import absltest

EMBEDDING_SIZE = 8


class ThreadLocalCursorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def test_each_thread_gets_own_cursor(self):
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 0, np.random.default_rng(0), EMBEDDING_SIZE
    )
    dep_id = db.insert_deployment(name='cursor_dep', project='cursor_project')
    rec_id = db.insert_recording(filename='cursor_test.wav', deployment_id=dep_id)
    db.commit()

    cursor_ids = {}
    cursor_ids_lock = threading.Lock()
    barrier = threading.Barrier(4)
    done_events = [threading.Event() for _ in range(4)]

    def get_cursor_id(thread_idx):
      cursor = db._get_cursor()
      cursor_id = id(cursor)
      with cursor_ids_lock:
        cursor_ids[thread_idx] = cursor_id
      barrier.wait()
      done_events[thread_idx].wait(timeout=2)

    threads = [threading.Thread(target=get_cursor_id, args=(i,), name=f'worker_{i}')
               for i in range(4)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=5)

    unique_cursors = set(cursor_ids.values())
    self.assertLen(unique_cursors, 4)

  def test_sequential_read_write_after_commit(self):
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 10, np.random.default_rng(42), EMBEDDING_SIZE
    )
    dep_id = db.insert_deployment(name='seq_dep', project='seq_project')
    rec_id = db.insert_recording(filename='seq.wav', deployment_id=dep_id)
    db.commit()

    embedding = np.random.default_rng(99).normal(size=EMBEDDING_SIZE).astype(np.float16)
    wid = db.insert_window(
        recording_id=rec_id, offsets=[0.0, 5.0], embedding=embedding
    )
    db.commit()

    got = db.get_window(wid)
    self.assertEqual(got.id, wid)

    ids = db.match_window_ids()
    self.assertIn(wid, ids)

  def test_commit_clears_cursor(self):
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 0, np.random.default_rng(0), EMBEDDING_SIZE
    )

    db.insert_deployment(name='cr_dep', project='cr_project')
    self.assertIsNotNone(db._thread_local.cursor)

    db.commit()
    self.assertIsNone(db._thread_local.cursor)

    db.insert_deployment(name='cr_dep2', project='cr_project2')
    self.assertIsNotNone(db._thread_local.cursor)

  def test_rollback_clears_cursor(self):
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 0, np.random.default_rng(0), EMBEDDING_SIZE
    )

    db.insert_deployment(name='rb_dep', project='rb_project')
    self.assertIsNotNone(db._thread_local.cursor)

    db.rollback()
    self.assertIsNone(db._thread_local.cursor)

  def test_cursor_recreated_after_commit(self):
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 0, np.random.default_rng(0), EMBEDDING_SIZE
    )

    dep_id = db.insert_deployment(name='re_dep', project='re_project')
    cursor_before = db._thread_local.cursor
    cursor_id_before = id(cursor_before)
    db.commit()

    db.insert_deployment(name='re_dep2', project='re_project2')
    cursor_after = db._thread_local.cursor
    cursor_id_after = id(cursor_after)

    self.assertIsNot(cursor_before, cursor_after)

  def test_thread_split_creates_independent_cursors(self):
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 10, np.random.default_rng(42), EMBEDDING_SIZE
    )
    split_db = db.thread_split()

    cursor_main = db._get_cursor()
    cursor_split = split_db._get_cursor()

    self.assertIsNot(cursor_main, cursor_split)

  def test_multiple_threads_sequential_access(self):
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 10, np.random.default_rng(42), EMBEDDING_SIZE
    )
    dep_id = db.insert_deployment(name='mt_dep', project='mt_project')
    rec_id = db.insert_recording(filename='mt.wav', deployment_id=dep_id)
    db.commit()

    errors = []

    def worker(thread_idx):
      try:
        embedding = np.random.default_rng(thread_idx).normal(
            size=EMBEDDING_SIZE
        ).astype(np.float16)
        wid = db.insert_window(
            recording_id=rec_id,
            offsets=[float(thread_idx * 10.0), float(thread_idx * 10.0 + 5.0)],
            embedding=embedding,
        )
        db.commit()
        got = db.get_window(wid)
        assert got.id == wid
      except Exception as e:
        errors.append(e)

    for i in range(4):
      t = threading.Thread(target=worker, args=(i,), name=f'w_{i}')
      t.start()
      t.join(timeout=10)

    self.assertEmpty(errors)
    ids = db.match_window_ids()
    self.assertLen(ids, 14)


if __name__ == '__main__':
  absltest.main()
