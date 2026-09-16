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


  def test_concurrent_commits_are_serialized(self):
    """Verify that concurrent commits do not interfere with each other.

    Multiple threads insert data and commit concurrently. The lock ensures
    that commits are serialized, preventing one thread's commit from
    interfering with another thread's in-flight transaction.
    """
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 10, np.random.default_rng(42), EMBEDDING_SIZE
    )
    dep_id = db.insert_deployment(name='cc_dep', project='cc_project')
    rec_id = db.insert_recording(filename='cc.wav', deployment_id=dep_id)
    db.commit()

    errors = []
    num_threads = 8
    barrier = threading.Barrier(num_threads)

    def worker(thread_idx):
      try:
        barrier.wait()
        for j in range(5):
          embedding = np.random.default_rng(thread_idx * 100 + j).normal(
              size=EMBEDDING_SIZE
          ).astype(np.float16)
          wid = db.insert_window(
              recording_id=rec_id,
              offsets=[float(thread_idx * 10.0 + j), float(thread_idx * 10.0 + j + 1.0)],
              embedding=embedding,
          )
          db.commit()
          got = db.get_window(wid)
          assert got.id == wid
      except Exception as e:
        errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(i,), name=f'cc_{i}')
        for i in range(num_threads)
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=30)

    self.assertEmpty(errors)
    ids = db.match_window_ids()
    self.assertLen(ids, 10 + num_threads * 5)

  def test_concurrent_commit_and_rollback(self):
    """Verify that concurrent commits and rollbacks are serialized.

    Some threads commit while others rollback. The lock ensures no
    cross-thread interference on the shared connection.
    """
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 10, np.random.default_rng(42), EMBEDDING_SIZE
    )
    dep_id = db.insert_deployment(name='cr_dep2', project='cr_project2')
    rec_id = db.insert_recording(filename='cr.wav', deployment_id=dep_id)
    db.commit()

    errors = []
    num_threads = 6
    barrier = threading.Barrier(num_threads)

    def committer(thread_idx):
      try:
        barrier.wait()
        for j in range(3):
          embedding = np.random.default_rng(thread_idx * 100 + j).normal(
              size=EMBEDDING_SIZE
          ).astype(np.float16)
          db.insert_window(
              recording_id=rec_id,
              offsets=[float(thread_idx * 10.0 + j), float(thread_idx * 10.0 + j + 1.0)],
              embedding=embedding,
          )
          db.commit()
      except Exception as e:
        errors.append(e)

    def rollbacker(thread_idx):
      try:
        barrier.wait()
        for _ in range(3):
          embedding = np.random.default_rng(thread_idx * 100).normal(
              size=EMBEDDING_SIZE
          ).astype(np.float16)
          db.insert_window(
              recording_id=rec_id,
              offsets=[900.0, 901.0],
              embedding=embedding,
          )
          db.rollback()
      except Exception as e:
        errors.append(e)

    threads = []
    for i in range(num_threads // 2):
      threads.append(threading.Thread(target=committer, args=(i,), name=f'c_{i}'))
      threads.append(threading.Thread(target=rollbacker, args=(i + 3,), name=f'r_{i}'))

    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=30)

    self.assertEmpty(errors)


if __name__ == '__main__':
  absltest.main()
