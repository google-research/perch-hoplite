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

"""Tests for EmbedWorker transaction commit cadence."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from perch_hoplite.agile import embed


class CommitCadenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('every_batch', 1, 3),
      ('every_two_batches', 2, 2),
      ('end_only', None, 1),
  )
  def test_embed_dataset_commit_cadence(self, commit_every_n_batches, expected):
    worker = object.__new__(embed.EmbedWorker)
    worker.timestamp_resolver = None
    worker.db = mock.MagicMock()
    worker.db.thread_split.return_value = mock.Mock()
    worker.audio_worker_threads = 1
    worker.window_size_s = 1.0
    worker.audio_sources = mock.Mock()
    worker.audio_sources.iterate_all_sources.return_value = [
        mock.Mock(file_id=f'file-{i}.wav', dataset_name='test') for i in range(5)
    ]
    worker.get_recording_timestamp = mock.Mock(return_value=None)

    with mock.patch.object(embed, 'process_source_id', return_value=None):
      worker.embed_dataset(
          batch_size=2,
          new_recordings=set(),
          commit_every_n_batches=commit_every_n_batches,
      )

    self.assertEqual(worker.db.commit.call_count, expected)

  def test_embed_dataset_rejects_invalid_commit_cadence(self):
    worker = object.__new__(embed.EmbedWorker)
    worker.timestamp_resolver = None

    with self.assertRaisesRegex(ValueError, 'must be positive or None'):
      worker.embed_dataset(commit_every_n_batches=0)

  def test_process_all_forwards_commit_cadence(self):
    worker = object.__new__(embed.EmbedWorker)
    worker.db = mock.MagicMock()
    worker.db.count_embeddings.return_value = 1
    worker.update_configs = mock.Mock()
    worker.add_deployments = mock.Mock()
    worker.add_recordings = mock.Mock(return_value={7})
    worker.add_annotations = mock.Mock()
    worker.embed_dataset = mock.Mock()

    worker.process_all(
        target_dataset_name='test',
        batch_size=4,
        handle_duplicates='skip',
        commit_every_n_batches=3,
    )

    worker.embed_dataset.assert_called_once_with(
        batch_size=4,
        handle_duplicates='skip',
        target_dataset_name='test',
        new_recordings={7},
        commit_every_n_batches=3,
    )


if __name__ == '__main__':
  absltest.main()
