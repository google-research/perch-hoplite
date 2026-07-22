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

"""Tests for embedding audio."""

import datetime
import os
import shutil
import tempfile

from ml_collections import config_dict
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.agile import timestamp_resolver
from perch_hoplite.agile.tests import test_utils
from perch_hoplite.db import db_loader

from absl.testing import absltest
from absl.testing import parameterized


class EmbedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()
    self.db_path = os.path.join(self.tempdir, 'test_db')

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  @parameterized.named_parameters(
      ('in_mem', 'in_mem'),
      ('sqlite_usearch', 'sqlite_usearch'),
  )
  def test_embed_worker(self, db_key):
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    test_utils.make_wav_files(self.tempdir, classes, filenames, file_len_s=6.0)

    # Create metadata files.
    with open(
        os.path.join(self.tempdir, 'hoplite_metadata_description.csv'), 'w'
    ) as f:
      f.write(
          'field_name,metadata_level,type,description\n'
          'deployment,deployment,str,Deployment identifier.\n'
          'habitat,deployment,str,Habitat type.\n'
          'recording,recording,str,Recording identifier.\n'
          'mic_type,recording,str,Microphone type.\n'
      )
    with open(
        os.path.join(self.tempdir, 'hoplite_deployments_metadata.csv'), 'w'
    ) as f:
      f.write('deployment,habitat\npos,forest\nneg,grassland\n')
    with open(
        os.path.join(self.tempdir, 'hoplite_recordings_metadata.csv'), 'w'
    ) as f:
      f.write(
          'recording,mic_type\n'
          'pos/foo.wav,MicA\n'
          'pos/bar.wav,MicA\n'
          'pos/baz.wav,MicB\n'
          'neg/foo.wav,MicA\n'
          'neg/bar.wav,MicB\n'
          'neg/baz.wav,MicB\n'
      )

    audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='test',
                base_path=self.tempdir,
                file_glob='*/*.wav',
                min_audio_len_s=0.0,
                target_sample_rate_hz=16000,
            ),
        )
    )

    placeholder_model_config = config_dict.ConfigDict()
    placeholder_model_config.embedding_size = 32
    placeholder_model_config.sample_rate = 16000
    model_config = embed.ModelConfig(
        model_key='placeholder_model',
        embedding_dim=32,
        model_config=placeholder_model_config,
    )

    with self.subTest('embedding'):
      embedding_dim = 32
      if db_key == 'in_mem':
        in_mem_db_config = config_dict.ConfigDict()
        in_mem_db_config.embedding_dim = embedding_dim
        db_config = db_loader.DBConfig(
            db_key='in_mem',
            db_config=in_mem_db_config,
        )
        db = db_config.load_db()
      elif db_key == 'sqlite_usearch':
        db_path = self.db_path + '_embedding'
        db = db_loader.create_new_usearch_db(
            db_path=db_path, embedding_dim=embedding_dim
        )
      else:
        raise ValueError(f'Unknown db_key: {db_key}')

      embed_worker = embed.EmbedWorker(
          audio_sources=audio_sources,
          model_config=model_config,
          db=db,
      )
      embed_worker.process_all()
      # The hop size is 1.0s and each file is 6.0s, so we get 6 embeddings
      # per file. There are six files, so we should get 36 embeddings.
      self.assertEqual(db.count_embeddings(), 36)
      embs = db.get_embeddings_batch(db.match_window_ids())
      self.assertEqual(embs.shape[-1], 32)
      self.assertLen(db.get_all_deployments(), 2)
      self.assertLen(db.get_all_recordings(), 6)

      # Check that metadata got attached to deployments and recordings.
      deployments = db.get_all_deployments()
      for d in deployments:
        if d.name == 'pos':
          self.assertEqual(d.habitat, 'forest')
        elif d.name == 'neg':
          self.assertEqual(d.habitat, 'grassland')
      recordings = db.get_all_recordings()
      for r in recordings:
        if r.filename == 'pos/foo.wav':
          self.assertEqual(r.mic_type, 'MicA')
        elif r.filename == 'pos/baz.wav':
          self.assertEqual(r.mic_type, 'MicB')

      # Check that the metadata is set correctly.
      got_md = db.get_metadata(key=None)
      self.assertIn('audio_sources', got_md)
      self.assertIn('model_config', got_md)

    with self.subTest('labels'):
      embedding_dim = 6
      if db_key == 'in_mem':
        in_mem_db_config = config_dict.ConfigDict()
        in_mem_db_config.embedding_dim = embedding_dim
        db_config = db_loader.DBConfig(
            db_key='in_mem',
            db_config=in_mem_db_config,
        )
        db = db_config.load_db()
      elif db_key == 'sqlite_usearch':
        db_path = self.db_path + '_labels'
        db = db_loader.create_new_usearch_db(
            db_path=db_path, embedding_dim=embedding_dim
        )
      else:
        raise ValueError(f'Unknown db_key: {db_key}')

      model_config.logits_key = 'label'
      model_config.logits_idxes = (1, 2, 3, 5, 8, 13)

      embed_worker = embed.EmbedWorker(
          audio_sources=audio_sources,
          model_config=model_config,
          db=db,
      )
      embed_worker.process_all()
      # The hop size is 1.0s and each file is 6.0s, so we get 6 embeddings
      # per file. There are six files, so we should get 36 embeddings.
      self.assertEqual(db.count_embeddings(), 36)
      embs = db.get_embeddings_batch(db.match_window_ids())
      # The placeholder model defaults to 128-dim'l outputs, but we only want
      # the channels specified in the logits_idxes.
      self.assertEqual(embs.shape[-1], 6)


class DuplicateHandlingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()
    self.db_path = os.path.join(self.tempdir, 'test_db')

    classes = ['pos']
    filenames = ['foo']
    test_utils.make_wav_files(self.tempdir, classes, filenames, file_len_s=2.0)

    self.audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='test',
                base_path=self.tempdir,
                file_glob='*/*.wav',
                min_audio_len_s=0.0,
                target_sample_rate_hz=16000,
            ),
        )
    )

    placeholder_model_config = config_dict.ConfigDict()
    placeholder_model_config.embedding_size = 32
    placeholder_model_config.sample_rate = 16000
    self.model_config = embed.ModelConfig(
        model_key='placeholder_model',
        embedding_dim=32,
        model_config=placeholder_model_config,
    )

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def test_duplicate_error(self):
    db = db_loader.create_new_usearch_db(db_path=self.db_path, embedding_dim=32)
    worker = embed.EmbedWorker(self.audio_sources, self.model_config, db)

    # First pass
    worker.process_all(handle_duplicates='allow')
    self.assertEqual(db.count_embeddings(), 2)  # 2s / 1s hop = 2

    # Second pass with error
    with self.assertRaisesRegex(ValueError, 'already exists'):
      worker.process_all(handle_duplicates='error')

  def test_duplicate_skip(self):
    db = db_loader.create_new_usearch_db(db_path=self.db_path, embedding_dim=32)
    worker = embed.EmbedWorker(self.audio_sources, self.model_config, db)

    worker.process_all(handle_duplicates='allow')
    self.assertEqual(db.count_embeddings(), 2)

    # Second pass with skip - should not add anything extra
    worker.process_all(handle_duplicates='skip')
    self.assertEqual(db.count_embeddings(), 2)

  def test_duplicate_overwrite(self):
    db = db_loader.create_new_usearch_db(db_path=self.db_path, embedding_dim=32)
    worker = embed.EmbedWorker(self.audio_sources, self.model_config, db)

    worker.process_all(handle_duplicates='allow')
    self.assertEqual(db.count_embeddings(), 2)

    # Second pass with overwrite - should remove and re-add.
    # Since it removes properly and then adds, count should still be 2.
    worker.process_all(handle_duplicates='overwrite')
    self.assertEqual(db.count_embeddings(), 2)

  def test_duplicate_allow(self):
    db = db_loader.create_new_usearch_db(db_path=self.db_path, embedding_dim=32)
    worker = embed.EmbedWorker(self.audio_sources, self.model_config, db)

    worker.process_all(handle_duplicates='allow')
    self.assertEqual(db.count_embeddings(), 2)
    worker.process_all(handle_duplicates='allow')
    self.assertEqual(db.count_embeddings(), 4)

  def test_timestamp_resolver(self):
    db = db_loader.create_new_usearch_db(db_path=self.db_path, embedding_dim=32)

    class ExampleTimestampResolver(timestamp_resolver.TimestampResolver):

      def get_filepath_timestamp(self, *args, **kwargs):
        return datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)

    worker = embed.EmbedWorker(
        self.audio_sources,
        self.model_config,
        db,
        timestamp_resolver=ExampleTimestampResolver(
            db=db, base_path=self.tempdir
        ),
    )
    worker.process_all(handle_duplicates='allow')
    self.assertEqual(db.count_embeddings(), 2)
    windows = db.get_all_windows()
    expected_ts = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    self.assertTrue(
        all(
            hasattr(w, 'timestamp')
            and (
                w.timestamp == expected_ts
                or (
                    isinstance(w.timestamp, str)
                    and datetime.datetime.fromisoformat(w.timestamp)
                    == expected_ts
                )
            )
            for w in windows
        )
    )

  def test_timestamp_file_pattern(self):
    tempdir = tempfile.mkdtemp()
    try:
      db = db_loader.create_new_usearch_db(
          db_path=os.path.join(tempdir, 'test_db'), embedding_dim=32
      )

      classes = ['pos']
      filenames = ['20231024_153000']
      test_utils.make_wav_files(tempdir, classes, filenames, file_len_s=2.0)

      audio_sources = source_info.AudioSources(
          audio_globs=(
              source_info.AudioSourceConfig(
                  dataset_name='test',
                  base_path=tempdir,
                  file_glob='*/*.wav',
                  min_audio_len_s=0.0,
                  target_sample_rate_hz=16000,
              ),
          )
      )

      worker = embed.EmbedWorker(
          audio_sources,
          self.model_config,
          db,
          timestamp_file_pattern='%Y%m%d_%H%M%S_pos',
      )
      worker.process_all(handle_duplicates='allow')

      recordings = db.get_all_recordings()
      self.assertLen(recordings, 1)
      recording = recordings[0]
      expected_recording_dt = datetime.datetime(
          2023, 10, 24, 15, 30, 0, tzinfo=datetime.timezone.utc
      )
      self.assertEqual(recording.datetime, expected_recording_dt)

      windows = db.get_all_windows()
      self.assertLen(windows, 2)

      def get_timestamp(w):
        ts = getattr(w, 'timestamp', None)
        if isinstance(ts, str):
          return datetime.datetime.fromisoformat(ts)
        return ts

      self.assertEqual(
          get_timestamp(windows[0]),
          datetime.datetime(
              2023, 10, 24, 15, 30, 0, tzinfo=datetime.timezone.utc
          ),
      )
      self.assertTrue(hasattr(windows[1], 'timestamp'))
      self.assertEqual(
          get_timestamp(windows[1]),
          datetime.datetime(
              2023, 10, 24, 15, 30, 1, tzinfo=datetime.timezone.utc
          ),
      )
    finally:
      shutil.rmtree(tempdir)


if __name__ == '__main__':
  absltest.main()
