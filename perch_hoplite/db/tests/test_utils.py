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

"""Utility functions for testing."""

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import interface
from perch_hoplite.db import multi_db_impl
from perch_hoplite.db import sqlite_usearch_impl

# DB types for testing.
DB_TYPES = (
    'in_mem',
    'sqlite_usearch',
    'multi_db_in_mem',
    'multi_db_sqlite_usearch',
)
DB_TYPE_NAMED_PAIRS = (
    ('in_mem-sqlite_usearch', 'in_mem', 'sqlite_usearch'),
    (
        'multi_db_in_mem-multi_db_sqlite_usearch',
        'multi_db_in_mem',
        'multi_db_sqlite_usearch',
    ),
)
PERSISTENT_DB_TYPES = ('sqlite_usearch', 'multi_db_sqlite_usearch')

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
    db = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=embedding_dim)
  elif db_type == 'sqlite_usearch':
    usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(embedding_dim)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_path=path, usearch_cfg=usearch_cfg
    )
  elif db_type == 'multi_db_in_mem':
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=embedding_dim)
    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=embedding_dim)
    db = multi_db_impl.MultiDBWrapper.create(dbs={'db0': db0, 'db1': db1})
  elif db_type == 'multi_db_sqlite_usearch':
    usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(embedding_dim)
    db0 = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_path=f'{path}_0', usearch_cfg=usearch_cfg
    )
    db1 = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_path=f'{path}_1', usearch_cfg=usearch_cfg
    )
    db = multi_db_impl.MultiDBWrapper.create(dbs={'db0': db0, 'db1': db1})
  else:
    raise ValueError(f'Unknown db type: {db_type}')
  # Insert a few embeddings...
  if fill_random:
    insert_random_embeddings(db, embedding_dim, num_embeddings, rng)
  config = config_dict.ConfigDict()
  config.embedding_dim = embedding_dim
  model_config = config_dict.ConfigDict()
  model_config.embedding_dim = embedding_dim
  model_config.model_name = 'fake_model'
  if isinstance(db, multi_db_impl.MultiDBWrapper):
    db.insert_metadata('db0/db_config', config)
    db.insert_metadata('db0/model_config', model_config)
    db.insert_metadata('db1/db_config', config)
    db.insert_metadata('db1/model_config', model_config)
  else:
    db.insert_metadata('db_config', config)
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

  projects = ('a', 'b', 'c')
  deployment_ids = []
  for project in projects:
    if isinstance(db, multi_db_impl.MultiDBWrapper):
      db.get_dbs()[0].insert_deployment(
          name=f'deployment_{project}', project=project
      )
    deployment_id = db.insert_deployment(
        name=f'deployment_{project}', project=project
    )
    deployment_ids.append(deployment_id)

  window_size_s = 5.0
  for _ in range(num_embeddings):
    deployment_id = rng.choice(deployment_ids).item()
    filename = ''.join(
        [str(a) for a in rng.choice(np_alpha, size=8, replace=False)]
    )
    recording_id = db.insert_recording(
        filename=filename, deployment_id=deployment_id
    )

    embedding = np.float32(rng.normal(size=emb_dim, loc=0, scale=0.1))
    offsets = rng.integers(0, 100, size=[1])
    offsets = [offsets[0], offsets[0] + window_size_s]
    db.insert_window(
        recording_id, offsets, embedding, handle_duplicates='allow'
    )
  db.commit()


def clone_embeddings(
    source_db: interface.HopliteDBInterface,
    target_db: interface.HopliteDBInterface,
):
  """Copy all embeddings to target_db and provide an id mapping."""

  # First, clone deployments and keep a map between source and target ids.
  deployment_id_mapping = {None: None}
  for deployment in source_db.get_all_deployments():
    kwargs = deployment.to_kwargs(skip=['id'])
    if isinstance(target_db, multi_db_impl.MultiDBWrapper):
      target_db.get_dbs()[0].insert_deployment(**kwargs)
    target_id = target_db.insert_deployment(**kwargs)
    deployment_id_mapping[deployment.id] = target_id

  # Second, clone recordings and keep a map between source and target ids.
  recording_id_mapping = {}
  for recording in source_db.get_all_recordings():
    target_id = target_db.insert_recording(
        deployment_id=deployment_id_mapping[recording.deployment_id],
        **recording.to_kwargs(skip=['id', 'deployment_id']),
    )
    recording_id_mapping[recording.id] = target_id

  # Finally, clone windows and keep a map between source and target ids.
  window_id_mapping = {}
  for window in source_db.get_all_windows():
    target_id = target_db.insert_window(
        recording_id=recording_id_mapping[window.recording_id],
        embedding=source_db.get_embedding(window.id),
        handle_duplicates='allow',
        **window.to_kwargs(skip=['id', 'embedding', 'recording_id']),
    )
    window_id_mapping[window.id] = target_id

  # Return the window id mapping.
  return window_id_mapping


def add_random_labels(
    db: interface.HopliteDBInterface,
    rng: np.random.Generator,
    unlabeled_prob: float = 0.5,
    positive_label_prob: float = 0.5,
    provenance: str = 'test',
):
  """Insert random labels for a subset of embeddings."""
  for idx in db.match_window_ids():
    if rng.random() < unlabeled_prob:
      continue
    if rng.random() < positive_label_prob:
      label_type = datatypes.LabelType.POSITIVE
    else:
      label_type = datatypes.LabelType.NEGATIVE
    window = db.get_window(idx)
    db.insert_annotation(
        recording_id=window.recording_id,
        offsets=window.offsets,
        label=str(rng.choice(CLASS_LABELS)),
        label_type=label_type,
        provenance=provenance,
        handle_duplicates='allow',
    )
  db.commit()
