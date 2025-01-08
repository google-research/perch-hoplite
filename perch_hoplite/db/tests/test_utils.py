# coding=utf-8
# Copyright 2024 The Perch Authors.
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

"""Utility functions for testing."""

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import interface
from perch_hoplite.db import sqlite_usearch_impl


CLASS_LABELS = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta')


def make_db(
    path: str,
    db_type: str,
    num_embeddings: int,
    rng: np.random.Generator,
    embedding_dim: int = 128,
    fill_random: bool = True,
) -> interface.HopliteDBInterface:
  """Create a test DB of the specified type."""
  if db_type == 'in_mem':
    db = in_mem_impl.InMemoryGraphSearchDB.create(
        embedding_dim=embedding_dim,
        max_size=num_embeddings,
    )
  elif db_type == 'sqlite_usearch':
    usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(embedding_dim)
    db = sqlite_usearch_impl.SQLiteUsearchDB.create(
        db_path=path, usearch_cfg=usearch_cfg
    )
  else:
    raise ValueError(f'Unknown db type: {db_type}')
  # Insert a few embeddings...
  if fill_random:
    insert_random_embeddings(db, embedding_dim, num_embeddings, rng)
  config = config_dict.ConfigDict()
  config.embedding_dim = embedding_dim
  db.insert_metadata('db_config', config)
  model_config = config_dict.ConfigDict()
  model_config.embedding_dim = embedding_dim
  model_config.model_name = 'fake_model'
  db.insert_metadata('model_config', model_config)
  db.commit()
  return db


def insert_random_embeddings(
    db: interface.HopliteDBInterface,
    emb_dim: int = 1280,
    num_embeddings: int = 1000,
    seed: int = 42,
):
  """Insert randomly generated embedding vectors into the DB."""
  rng = np.random.default_rng(seed=seed)
  np_alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
  dataset_names = ('a', 'b', 'c')
  for _ in range(num_embeddings):
    embedding = np.float32(rng.normal(size=emb_dim, loc=0, scale=0.1))
    dataset_name = rng.choice(dataset_names)
    source_name = ''.join(
        [str(a) for a in rng.choice(np_alpha, size=8, replace=False)]
    )
    offsets = rng.integers(0, 100, size=[1])
    source = interface.EmbeddingSource(dataset_name, source_name, offsets)
    db.insert_embedding(embedding, source)
  db.commit()


def clone_embeddings(
    source_db: interface.HopliteDBInterface,
    target_db: interface.HopliteDBInterface,
):
  """Copy all embeddings to target_db and provide an id mapping."""
  id_mapping = {}
  for source_id in source_db.get_embedding_ids():
    id_mapping[source_id] = target_db.insert_embedding(
        source_db.get_embedding(source_id),
        source_db.get_embedding_source(source_id),
    )
  return id_mapping


def add_random_labels(
    db: interface.HopliteDBInterface,
    rng: np.random.Generator,
    unlabeled_prob: float = 0.5,
    positive_label_prob: float = 0.5,
    provenance: str = 'test',
):
  """Insert random labels for a subset of embeddings."""
  for idx in db.get_embedding_ids():
    if rng.random() < unlabeled_prob:
      continue
    if rng.random() < positive_label_prob:
      label_type = interface.LabelType.POSITIVE
    else:
      label_type = interface.LabelType.NEGATIVE
    label = interface.Label(
        embedding_id=idx,
        label=str(rng.choice(CLASS_LABELS)),
        type=label_type,
        provenance=provenance,
    )
    db.insert_label(label)
  db.commit()
