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

from ml_collections import config_dict
import numpy as np
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.agile import timestamp_resolver
from perch_hoplite.agile.tests import test_utils
from perch_hoplite.db import datatypes
from perch_hoplite.db import db_loader
from perch_hoplite.zoo import placeholder_model
from perch_hoplite.zoo import zoo_interface

from absl.testing import absltest
from absl.testing import parameterized


class EmbedTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = self.create_tempdir().full_path
    self.db_path = os.path.join(self.tempdir, 'test_db')

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
    self.tempdir = self.create_tempdir().full_path
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
    tempdir = self.create_tempdir().full_path
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

  def test_deployment_from_recording_metadata(self):
    tempdir = self.create_tempdir().full_path
    db = db_loader.create_new_usearch_db(
        db_path=os.path.join(tempdir, 'test_db'), embedding_dim=32
    )
    test_utils.make_wav_files(
        tempdir, ['audio'], ['rec1', 'rec2'], file_len_s=2.0
    )
    with open(
        os.path.join(tempdir, 'hoplite_metadata_description.csv'), 'w'
    ) as f:
      f.write(
          'field_name,metadata_level,type,description\n'
          'deployment,deployment,str,Deployment identifier.\n'
          'habitat,deployment,str,Habitat type.\n'
          'recording,recording,str,Recording identifier.\n'
          'deployment_name,recording,str,Deployment identifier.\n'
      )
    with open(
        os.path.join(tempdir, 'hoplite_deployments_metadata.csv'), 'w'
    ) as f:
      f.write('deployment,habitat\nsite_a,forest\nsite_b,wetland\n')
    with open(
        os.path.join(tempdir, 'hoplite_recordings_metadata.csv'), 'w'
    ) as f:
      f.write(
          'recording,deployment_name\n'
          'audio/rec1_audio.wav,site_a\n'
          'audio/rec2_audio.wav,site_b\n'
      )
    with open(os.path.join(tempdir, 'hoplite_annotations.csv'), 'w') as f:
      f.write(
          'recording,label,start_offset_s,end_offset_s,label_type\n'
          'audio/rec1_audio.wav,sp1,0.0,1.0,POSITIVE\n'
      )

    audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='test',
                base_path=tempdir,
                file_glob='audio/*.wav',
                min_audio_len_s=0.0,
                target_sample_rate_hz=16000,
            ),
        )
    )
    worker = embed.EmbedWorker(audio_sources, self.model_config, db)
    worker.process_all()

    deployments = {d.name: d for d in db.get_all_deployments()}
    self.assertSameElements(deployments.keys(), ['site_a', 'site_b'])
    self.assertEqual(deployments['site_a'].habitat, 'forest')
    self.assertEqual(deployments['site_b'].habitat, 'wetland')
    recordings = {r.filename: r for r in db.get_all_recordings()}
    self.assertEqual(
        recordings['audio/rec1_audio.wav'].deployment_id,
        deployments['site_a'].id,
    )
    self.assertEqual(
        recordings['audio/rec2_audio.wav'].deployment_id,
        deployments['site_b'].id,
    )
    self.assertLen(db.get_all_annotations(), 1)


class DetectionsAndOptionalEmbeddingsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = self.create_tempdir().full_path
    self.db_path = os.path.join(self.tempdir, 'test_db')
    test_utils.make_wav_files(self.tempdir, ['pos'], ['foo'], file_len_s=5.0)
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
    placeholder_cfg = config_dict.ConfigDict()
    placeholder_cfg.embedding_size = 32
    placeholder_cfg.sample_rate = 16000
    self.model_config = embed.ModelConfig(
        model_key='placeholder_model',
        embedding_dim=32,
        model_config=placeholder_cfg,
    )

  def _make_db(self, db_key: str, suffix: str = ''):
    if db_key == 'in_mem':
      in_mem_cfg = config_dict.ConfigDict({'embedding_dim': 32})
      return db_loader.DBConfig('in_mem', in_mem_cfg).load_db()
    return db_loader.create_new_usearch_db(
        db_path=self.db_path + suffix, embedding_dim=32
    )

  def _make_custom_model(self):
    base_model = placeholder_model.PlaceholderModel(
        sample_rate=16000, embedding_size=32
    )
    cls_a, cls_b, cls_c = base_model.class_list.classes[:3]

    class FakeDetectionModel(placeholder_model.PlaceholderModel):

      def embed(
          self, audio_array: np.ndarray
      ) -> zoo_interface.InferenceOutputs:
        out = super().embed(audio_array)
        logits = np.full_like(out.logits['label'], -5.0)
        # cls_a (idx 0): adjacent detections in frames 0, 1 (scores 1.5, 3.2),
        # gap at frame 2, single detection at frame 4 (score 2.1).
        logits[0, 0] = 1.5
        logits[1, 0] = 3.2
        logits[4, 0] = 2.1
        # cls_b (idx 1): single detection at frame 2 (score 1.8).
        logits[2, 1] = 1.8
        # cls_c (idx 2): high score at frame 0, but excluded from subset list.
        logits[0, 2] = 9.0
        out.logits['label'] = logits
        return out

    return FakeDetectionModel(sample_rate=16000, embedding_size=32), (
        cls_a,
        cls_b,
        cls_c,
    )

  @parameterized.named_parameters(
      ('in_mem', 'in_mem'),
      ('sqlite_usearch', 'sqlite_usearch'),
  )
  def test_detections_merging_and_subset_class_list(self, db_key):
    db = self._make_db(db_key, '_det')
    model, (cls_a, cls_b, _) = self._make_custom_model()
    worker = embed.EmbedWorker(
        audio_sources=self.audio_sources,
        model_config=self.model_config,
        db=db,
        embedding_model=model,
        class_list=[cls_a, cls_b],
        min_detection_score=1.0,
    )
    worker.process_all()

    self.assertEqual(db.count_embeddings(), 5)
    annos = sorted(
        db.get_all_annotations(), key=lambda a: (a.label, a.offsets[0])
    )
    self.assertLen(annos, 3)

    ann_a1 = [a for a in annos if a.label == cls_a][0]
    ann_a2 = [a for a in annos if a.label == cls_a][1]
    ann_b = [a for a in annos if a.label == cls_b][0]

    np.testing.assert_allclose(ann_a1.offsets, [0.0, 2.0])
    self.assertAlmostEqual(ann_a1.model_score, 3.2, places=5)
    self.assertEqual(ann_a1.provenance, 'placeholder_model')
    self.assertEqual(ann_a1.label_type, datatypes.LabelType.POSITIVE)

    np.testing.assert_allclose(ann_a2.offsets, [4.0, 5.0])
    self.assertAlmostEqual(ann_a2.model_score, 2.1, places=5)

    np.testing.assert_allclose(ann_b.offsets, [2.0, 3.0])
    self.assertAlmostEqual(ann_b.model_score, 1.8, places=5)

  @parameterized.named_parameters(
      ('in_mem', 'in_mem'),
      ('sqlite_usearch', 'sqlite_usearch'),
  )
  def test_write_embeddings_false(self, db_key):
    db = self._make_db(db_key, '_no_emb')
    model, (cls_a, cls_b, _) = self._make_custom_model()
    worker = embed.EmbedWorker(
        audio_sources=self.audio_sources,
        model_config=self.model_config,
        db=db,
        embedding_model=model,
        write_embeddings=False,
        class_list=(cls_a, cls_b),
        min_detection_score=1.0,
    )
    worker.process_all()

    self.assertEqual(db.count_embeddings(), 0)
    self.assertEmpty(db.get_all_windows())
    self.assertLen(db.get_all_deployments(), 1)
    self.assertLen(db.get_all_recordings(), 1)
    self.assertLen(db.get_all_annotations(), 3)

    # Second pass with 'error' should still detect existing deployments/recordings.
    with self.assertRaisesRegex(ValueError, 'already exists'):
      worker.process_all(handle_duplicates='error')

    # Second pass with 'skip' should not duplicate annotations.
    worker.process_all(handle_duplicates='skip')
    self.assertLen(db.get_all_annotations(), 3)

  def test_invalid_class_in_class_list_raises(self):
    db = self._make_db('in_mem')
    model, _ = self._make_custom_model()
    with self.assertRaisesRegex(ValueError, 'Classes not found'):
      embed.EmbedWorker(
          audio_sources=self.audio_sources,
          model_config=self.model_config,
          db=db,
          embedding_model=model,
          class_list=['nonexistent_species_xyz'],
          min_detection_score=1.0,
      )


if __name__ == '__main__':
  absltest.main()
