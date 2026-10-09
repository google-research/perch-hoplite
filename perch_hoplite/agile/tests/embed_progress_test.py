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

"""Tests for progress reporting during lazily sharded audio embedding."""

import dataclasses
import threading
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import numpy as np
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info


class _FakeDB:

  def __init__(self):
    self.windows = []
    self.commits = 0

  def thread_split(self):
    return self

  def insert_windows_batch(self, windows, embeddings, handle_duplicates):
    self.windows.append((windows, embeddings, handle_duplicates))

  def commit(self):
    self.commits += 1

  def count_embeddings(self):
    return len(self.windows)


def _source(number):
  return source_info.SourceId(
      dataset_name='test',
      file_id='site/audio.wav',
      offset_s=float(number),
      shard_len_s=1.0,
      filepath='/unused/audio.wav',
      sample_rate_hz=16000,
  )


class EmbeddingProgressTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.db = _FakeDB()
    self.sources = [_source(i) for i in range(5)]
    # No model or actual recording is required to exercise the embedding
    # writer and per-segment progress callbacks.
    self.worker = object.__new__(embed.EmbedWorker)
    self.worker.db = self.db
    self.worker.audio_sources = SimpleNamespace(
        iterate_all_sources=lambda name: iter(self.sources)
    )
    self.worker.audio_worker_threads = 2
    self.worker.timestamp_resolver = None
    self.worker.window_size_s = 1.0
    self.worker.get_recording_timestamp = lambda *args: None
    self.worker._get_or_insert_deployment_id = lambda *args: 7
    self.worker._get_or_insert_recording_id = lambda *args: (12, False)
    self.process_patch = mock.patch.object(
        embed, 'process_source_id', side_effect=self._process_source
    )
    self.process_patch.start()
    self.addCleanup(self.process_patch.stop)

  def _process_source(
      self, _state, current_source, _window_size, _recording_timestamp
  ):
    if current_source.offset_s == 2:
      return None
    return (
        [current_source, current_source],
        [[0.0, 0.5], [0.5, 1.0]],
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        [None, None],
    )

  def test_progress_updates_for_all_segments_including_skipped(self):
    updates = []
    thread_ids = []

    def collect(status):
      thread_ids.append(threading.get_ident())
      updates.append(status)

    self.worker.embed_dataset(
        batch_size=2,
        new_recordings={12},
        progress_callback=collect,
    )
    self.assertLen(updates, 5)
    self.assertEqual([u.processed_segments for u in updates], [1, 2, 3, 4, 5])
    self.assertEqual([u.embedded_segments for u in updates], [1, 2, 2, 3, 4])
    self.assertEqual([u.generated_embeddings for u in updates], [2, 4, 4, 6, 8])
    self.assertEqual([u.source_id.offset_s for u in updates], [0, 1, 2, 3, 4])
    self.assertEqual(thread_ids, [threading.get_ident()] * 5)
    self.assertLen(self.db.windows, 4)
    self.assertEqual(self.db.commits, 1)

  def test_status_is_immutable(self):
    statuses = []
    self.worker.embed_dataset(
        batch_size=2,
        new_recordings={12},
        progress_callback=statuses.append,
    )
    with self.assertRaises(dataclasses.FrozenInstanceError):
      statuses[0].processed_segments = 100

  def test_progress_callback_is_optional(self):
    self.worker.embed_dataset(batch_size=2, new_recordings={12})
    self.assertLen(self.db.windows, 4)
    self.assertEqual(self.db.commits, 1)

  def test_empty_source_iterator_produces_no_status(self):
    self.sources = []
    statuses = []
    self.worker.embed_dataset(
        new_recordings={12}, progress_callback=statuses.append
    )
    self.assertEmpty(statuses)
    self.assertEmpty(self.db.windows)
    self.assertEqual(self.db.commits, 1)

  def test_progress_bar_tracks_segments_without_known_total(self):
    bars = []

    class Bar:

      def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = []
        bars.append(self)

      def __enter__(self):
        return self

      def __exit__(self, *_):
        return None

      def update(self, count):
        self.updates.append(count)

    with mock.patch.object(embed.tqdm, 'tqdm', Bar):
      self.worker.embed_dataset(batch_size=3, new_recordings={12})
    self.assertLen(bars, 1)
    self.assertEqual(bars[0].kwargs['unit'], 'segment')
    self.assertNotIn('total', bars[0].kwargs)
    self.assertEqual(bars[0].updates, [1, 1, 1, 1, 1])

  def test_progress_callback_error_is_not_silently_ignored(self):
    def fail(_status):
      raise RuntimeError('stop embedding')

    with self.assertRaisesRegex(RuntimeError, 'stop embedding'):
      self.worker.embed_dataset(
          batch_size=3, new_recordings={12}, progress_callback=fail
      )
    self.assertEqual(self.db.commits, 0)

  def test_process_all_forwards_progress_callback(self):
    self.worker.update_configs = mock.Mock()
    self.worker.add_deployments = mock.Mock()
    self.worker.add_recordings = mock.Mock(return_value={12})
    self.worker.add_annotations = mock.Mock()
    self.worker.embed_dataset = mock.Mock()
    callback = mock.Mock()

    self.worker.process_all(
        target_dataset_name='test',
        batch_size=7,
        handle_duplicates='skip',
        progress_callback=callback,
    )
    self.worker.embed_dataset.assert_called_once_with(
        batch_size=7,
        handle_duplicates='allow',
        target_dataset_name='test',
        new_recordings={12},
        progress_callback=callback,
    )

  def test_embedding_failure_does_not_report_success(self):
    original_process = self._process_source

    def fail_second(state, source, duration, timestamp):
      if source.offset_s == 1:
        raise ValueError('model failed')
      return original_process(state, source, duration, timestamp)

    statuses = []
    with mock.patch.object(embed, 'process_source_id', side_effect=fail_second):
      with self.assertRaisesRegex(ValueError, 'model failed'):
        self.worker.embed_dataset(
            batch_size=3,
            new_recordings={12},
            progress_callback=statuses.append,
        )
    self.assertLen(statuses, 1)
    self.assertEqual(statuses[0].processed_segments, 1)
    self.assertEqual(self.db.commits, 0)


if __name__ == '__main__':
  absltest.main()
