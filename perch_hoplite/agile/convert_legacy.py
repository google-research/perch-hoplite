# coding=utf-8
# Copyright 2025 The Perch Authors.
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

"""Conversion for TFRecord embeddings to Hoplite DB."""

from collections.abc import Sequence
import json

from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.db import db_loader
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import sqlite_usearch_impl
import tensorflow as tf
import tqdm


def convert_tfrecords(
    embeddings_path: str,
    db_type: str,
    dataset_name: str,
    max_count: int = -1,
    **kwargs,
):
  """Convert a TFRecord embeddings dataset to a Hoplite DB."""
  ds = create_embeddings_dataset(
      embeddings_path,
      'embeddings-*',
  )
  # Peek at one embedding to get the embedding dimension.
  for ex in ds.as_numpy_iterator():
    emb_dim = ex['embedding'].shape[-1]
    break
  else:
    raise ValueError('No embeddings found.')

  if db_type == 'sqlite_usearch':
    db_path = kwargs['db_path']
    if (epath.Path(db_path) / sqlite_usearch_impl.HOPLITE_FILENAME).exists():
      raise ValueError(f'DB path {db_path} already exists.')
    db = db_loader.create_new_usearch_db(db_path, emb_dim)
  elif db_type == 'in_mem':
    db = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=emb_dim)
  else:
    raise ValueError(f'Unknown db type: {db_type}')

  # Convert embedding config to new format and insert into the DB.
  legacy_config = load_embedding_config(embeddings_path)
  model_config = embed.ModelConfig(
      model_key=legacy_config.embed_fn_config.model_key,
      embedding_dim=emb_dim,
      model_config=legacy_config.embed_fn_config.model_config,
  )
  file_id_depth = legacy_config.embed_fn_config['file_id_depth']
  audio_globs = []
  for i, glob in enumerate(legacy_config.source_file_patterns):
    base_path, file_glob = glob.split('/')[-file_id_depth - 1 :]
    if i > 0:
      partial_dataset_name = f'{dataset_name}_{i}'
    else:
      partial_dataset_name = dataset_name
    audio_globs.append(
        source_info.AudioSourceConfig(
            dataset_name=partial_dataset_name,
            base_path=base_path,
            file_glob=file_glob,
            min_audio_len_s=legacy_config.embed_fn_config.min_audio_s,
            target_sample_rate_hz=legacy_config.embed_fn_config.get(
                'target_sample_rate_hz', -2
            ),
        )
    )

  audio_sources = source_info.AudioSources(audio_globs=tuple(audio_globs))
  db.insert_metadata('legacy_config', legacy_config)
  db.insert_metadata('audio_sources', audio_sources.to_config_dict())
  db.insert_metadata('model_config', model_config.to_config_dict())
  hop_size_s = model_config.model_config.hop_size_s
  window_size_s = model_config.model_config.window_size_s

  deployment_id = db.insert_deployment(name=dataset_name, project=dataset_name)
  recording_id_map = {}
  for ex in tqdm.tqdm(ds.as_numpy_iterator()):
    embs = ex['embedding']
    flat_embeddings = np.reshape(embs, [-1, embs.shape[-1]])
    file_id = str(ex['filename'], 'utf8')
    offset_s = ex['timestamp_s'].item()
    if max_count > 0 and db.count_embeddings() >= max_count:
      break
    # Long files were often split into shorter chunks during inference; ensure
    # consistent assignment of windows to recordings.
    if file_id in recording_id_map:
      recording_id = recording_id_map[file_id]
    else:
      recording_id = db.insert_recording(
          filename=file_id, deployment_id=deployment_id
      )
      recording_id_map[file_id] = recording_id
    windows_batch = [
        {
            'recording_id': recording_id,
            'offsets': [
                offset_s + hop_size_s * i,
                offset_s + hop_size_s * i + window_size_s,
            ],
        }
        for i in range(flat_embeddings.shape[0])
    ]
    db.insert_windows_batch(windows_batch, flat_embeddings)
    if max_count > 0 and db.count_embeddings() >= max_count:
      break
  db.commit()
  num_embeddings = db.count_embeddings()
  print('\n\nTotal embeddings : ', num_embeddings)
  hours_equiv = num_embeddings / 60 / 60 * hop_size_s
  print(f'\n\nHours of audio equivalent : {hours_equiv:.2f}')
  return db


# Functionality ported from chirp/inference/embed_lib.py and tf_examples.py.


def load_embedding_config(embeddings_path, filename: str = 'config.json'):
  """Loads the configuration to generate unlabeled embeddings."""
  embeddings_path = epath.Path(embeddings_path)
  with (embeddings_path / filename).open() as f:
    embedding_config = config_dict.ConfigDict(json.loads(f.read()))
  return embedding_config


# Feature keys.
FILE_NAME = 'filename'
TIMESTAMP_S = 'timestamp_s'
EMBEDDING = 'embedding'
EMBEDDING_SHAPE = 'embedding_shape'
LOGITS = 'logits'
SEPARATED_AUDIO = 'separated_audio'
SEPARATED_AUDIO_SHAPE = 'separated_audio_shape'
RAW_AUDIO = 'raw_audio'
RAW_AUDIO_SHAPE = 'raw_audio_shape'
FRONTEND = 'frontend'
FRONTEND_SHAPE = 'frontend_shape'


def get_feature_description(logit_names: Sequence[str] | None = None):
  """Create a feature description for the TFExamples.

  Each tensor feature includes both a serialized tensor and a 'shape' feature.
  The tensor feature can be parsed with tf.io.parse_tensor, and then reshaped
  according to the shape feature.

  Args:
    logit_names: Name of logit features included in the examples.

  Returns:
    Feature description dict for parsing TF Example protos.
  """
  feature_description = {
      FILE_NAME: tf.io.FixedLenFeature([], tf.string),
      TIMESTAMP_S: tf.io.FixedLenFeature([], tf.float32),
      EMBEDDING: tf.io.FixedLenFeature([], tf.string, default_value=''),
      EMBEDDING_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
      SEPARATED_AUDIO: tf.io.FixedLenFeature([], tf.string, default_value=''),
      SEPARATED_AUDIO_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
      FRONTEND: tf.io.FixedLenFeature([], tf.string, default_value=''),
      FRONTEND_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
      RAW_AUDIO: tf.io.FixedLenFeature([], tf.string, default_value=''),
      RAW_AUDIO_SHAPE: tf.io.FixedLenSequenceFeature(
          [], tf.int64, allow_missing=True
      ),
  }
  if logit_names is not None:
    for logit_name in logit_names:
      feature_description[logit_name] = tf.io.FixedLenFeature(
          [], tf.string, default_value=''
      )
      feature_description[f'{logit_name}_shape'] = (
          tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
      )
  return feature_description


def get_example_parser(
    logit_names: Sequence[str] | None = None, tensor_dtype: str = 'float32'
):
  """Create a parser for decoding inference library TFExamples."""
  features = get_feature_description(logit_names=logit_names)

  def _parser(ex):
    ex = tf.io.parse_single_example(ex, features)
    tensor_keys = [EMBEDDING, SEPARATED_AUDIO, RAW_AUDIO, FRONTEND]
    if logit_names is not None:
      tensor_keys.extend(logit_names)
    for key in tensor_keys:
      # Note that we can't use implicit truthiness for string tensors.
      # We are also required to have the same tensor structure and dtype in
      # both conditional branches. So we use an empty tensor when no
      # data is present to parse.
      if ex[key] != tf.constant(b'', dtype=tf.string):
        ex[key] = tf.io.parse_tensor(ex[key], out_type=tensor_dtype)
      else:
        ex[key] = tf.zeros_like([], dtype=tensor_dtype)
    return ex

  return _parser


def create_embeddings_dataset(
    embeddings_dir,
    file_glob: str = '*',
    prefetch: int = 128,
    logit_names: Sequence[str] | None = None,
    tensor_dtype: str = 'float32',
    shuffle_files: bool = False,
):
  """Create a TF Dataset of the embeddings."""
  embeddings_dir = epath.Path(embeddings_dir)
  embeddings_files = [fn.as_posix() for fn in embeddings_dir.glob(file_glob)]
  if shuffle_files:
    np.random.shuffle(embeddings_files)
  ds = tf.data.TFRecordDataset(
      embeddings_files, num_parallel_reads=tf.data.AUTOTUNE
  )

  parser = get_example_parser(
      logit_names=logit_names, tensor_dtype=tensor_dtype
  )
  ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.prefetch(prefetch)
  return ds
