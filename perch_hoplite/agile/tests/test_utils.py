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

"""Utilities for testing agile modeling functionality."""

import os
import shutil

from ml_collections import config_dict
import numpy as np
from perch_hoplite.agile import classifier
from perch_hoplite.db import datatypes
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.db.tests import test_utils as db_test_utils
from scipy.io import wavfile


def make_wav_files(
    base_path, classes, filenames, file_len_s=1.0, sample_rate_hz=16000
):
  """Create a pile of wav files in a directory structure."""
  rng = np.random.default_rng(seed=42)
  for subdir in classes:
    subdir_path = os.path.join(base_path, subdir)
    os.mkdir(subdir_path)
    for filename in filenames:
      with open(
          os.path.join(subdir_path, f'{filename}_{subdir}.wav'), 'wb'
      ) as f:
        noise = rng.normal(scale=0.2, size=int(file_len_s * sample_rate_hz))
        wavfile.write(f, sample_rate_hz, noise)
  audio_glob = os.path.join(base_path, '*/*.wav')
  return audio_glob


def make_mix_embeddings(
    rng: np.random.Generator,
    embedding_dim: int,
    mu_diff: float,
    pos_sigma: float,
    neg_sigma: float,
    pos_pi: float,
    n: int,
):
  """Create a Gaussian mixture of positive and negative embeddings.

  Args:
    rng: Random number generator.
    embedding_dim: Dimension of the embeddings.
    mu_diff: Distance between means of positive and negative data.
    pos_sigma: Standard deviation of the positive data.
    neg_sigma: Standard deviation of the negative data.
    pos_pi: Proportion of positive data.
    n: Number of samples to create.

  Returns:
    A tuple of data and labels.
  """
  major_axis = rng.normal(size=embedding_dim)
  major_axis /= np.linalg.norm(major_axis)
  pos_mu = major_axis * mu_diff / 2
  neg_mu = major_axis * -mu_diff / 2
  pos_sigma = np.eye(embedding_dim) * pos_sigma
  neg_sigma = np.eye(embedding_dim) * neg_sigma
  pos_data = rng.multivariate_normal(pos_mu, pos_sigma, size=int(n * pos_pi))
  neg_data = rng.multivariate_normal(
      neg_mu, neg_sigma, size=n - pos_data.shape[0]
  )
  data = np.concat([pos_data, neg_data])
  labels = np.concat(
      [np.ones([pos_data.shape[0]]), np.zeros([neg_data.shape[0]])]
  )
  ideal_classifier = classifier.LinearClassifier(
      beta=major_axis[:, np.newaxis],
      beta_bias=np.array([0.0]),
      classes=('test_class',),
      embedding_model_config=None,
  )
  return data, labels, ideal_classifier


def call_density_simulation(
    num_windows: int = 8192,
    embedding_dim: int = 2,
    rng_seed: int = 42,
    mu_diff: float = 2.0,
    pos_sigma: float = 1.0,
    neg_sigma: float = 1.0,
    pos_pi: float = 0.07,
):
  """Creates a test database for call density estimation."""
  if os.path.exists('/tmp/test_db'):
    shutil.rmtree('/tmp/test_db')
  rng = np.random.default_rng(rng_seed)
  db = sqlite_usearch_impl.SQLiteUSearchDB.create(
      db_path='/tmp/test_db',
      usearch_cfg=config_dict.create(
          dtype='float16',
          embedding_dim=embedding_dim,
          metric_name='IP',
          expansion_add=256,
          expansion_search=128,
      ),
  )
  # db = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=embedding_dim)
  data, labels, ideal_classifier = make_mix_embeddings(
      rng, embedding_dim, mu_diff, pos_sigma, neg_sigma, pos_pi, num_windows
  )
  deployment_id = db.insert_deployment(name='test', project='test')
  recording_id = db.insert_recording(
      filename='test', deployment_id=deployment_id
  )
  gt_labels_dict = {}
  for i, (emb, lbl) in enumerate(zip(data, labels)):
    idx = db.insert_window(
        recording_id=recording_id,
        embedding=emb,
        offsets=[5.0 * i, 5.0 * i + 5.0],
        handle_duplicates='allow',
    )
    if lbl == 1:
      gt_labels_dict[idx] = datatypes.LabelType.POSITIVE
    else:
      gt_labels_dict[idx] = datatypes.LabelType.NEGATIVE

  return db, ideal_classifier, gt_labels_dict
