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

"""Database configuration and constructor."""

import dataclasses

from etils import epath
from ml_collections import config_dict
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import interface
from perch_hoplite.db import sqlite_usearch_impl
import tqdm


@dataclasses.dataclass
class DBConfig(interface.EmbeddingMetadata):
  """Configuration for embedding database.

  Attributes:
    db_key: Key for the database implementation to use.
    db_config: Configuration for the database implementation.
  """

  db_key: str
  db_config: config_dict.ConfigDict

  def load_db(self) -> interface.HopliteDBInterface:
    """Load the database from the specified path."""
    if self.db_key == 'sqlite_usearch':
      return sqlite_usearch_impl.SQLiteUsearchDB.create(**self.db_config)
    elif self.db_key == 'in_mem':
      return in_mem_impl.InMemoryGraphSearchDB.create(**self.db_config)
    else:
      raise ValueError(f'Unknown db_key: {self.db_key}')


def duplicate_db(
    source_db: interface.HopliteDBInterface,
    target_db_key: str,
    target_db_config: config_dict.ConfigDict,
):
  """Create a new DB and copy all data in source_db into it."""
  target_db = DBConfig(target_db_key, target_db_config).load_db()
  target_db.commit()

  # Check that the target_db is empty. If not, we'll have to do something more
  # sophisticated.
  if target_db.get_embedding_ids().shape[0] > 0:
    raise ValueError('Target DB is not empty.')

  # Clone the embeddings and get the id mapping.
  # This automatically and seamlessly clones the source info as a
  # side effect.
  emb_id_mapping = {}
  for idx in tqdm.tqdm(source_db.get_embedding_ids()):
    emb = source_db.get_embedding(idx)
    source = source_db.get_embedding_source(idx)
    target_id = target_db.insert_embedding(emb, source)
    emb_id_mapping[idx] = target_id

  target_db.commit()

  # Clone labels.
  for idx, target_idx in tqdm.tqdm(emb_id_mapping.items()):
    for lbl in source_db.get_labels(idx):
      target_label = interface.Label(
          embedding_id=target_idx,
          label=lbl.label,
          provenance=lbl.provenance,
          type=lbl.type,
      )
      target_db.insert_label(target_label)
  target_db.commit()

  # Clone the KV store, replacing the DBConfig only.
  metadata = source_db.get_metadata(key=None)
  for k, v in metadata.items():
    if k == 'db_config':
      continue
    target_db.insert_metadata(k, v)
  target_db.insert_metadata('db_config', target_db_config)
  target_db.commit()

  return target_db, emb_id_mapping


def create_new_usearch_db(
    db_path: str,
    embedding_dim: int,
) -> sqlite_usearch_impl.SQLiteUsearchDB:
  """Create a new usearch DB with the given path and embedding dimension."""
  epath.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
  usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(embedding_dim)
  return sqlite_usearch_impl.SQLiteUsearchDB.create(
      db_path=db_path, usearch_cfg=usearch_cfg
  )
