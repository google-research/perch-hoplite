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

"""Parquet database implementation.

This implementation uses Parquet files for persistent storage.
Metadata tables (deployments, recordings, windows, annotations) are loaded
into in-memory pandas DataFrames at initialization for efficient filtering and
access. Embeddings are stored in a partitioned Parquet dataset on disk
(in the `embeddings/` subdirectory) to handle large datasets that may not fit
in memory.

Embeddings are only loaded from disk when explicitly requested via
`get_embedding()` or `get_embeddings_batch()`. When new embeddings are
inserted via `insert_window()`, they are buffered in a temporary in-memory
DataFrame (`_new_embeddings`). This buffer is flushed to the on-disk
Parquet dataset only when `commit()` is called, using
`pyarrow.parquet.write_to_dataset` to append to the dataset.

This design aims to balance memory efficiency for large embedding collections
with the performance benefits of in-memory filtering for metadata queries.
Search functionality is provided via brute-force search.
"""

import collections
from collections.abc import Sequence
import dataclasses
import datetime as dt
import itertools
import json
from typing import Any, Literal

from absl import logging
from etils import epath
from ml_collections import config_dict
import numpy as np
import pandas as pd
from perch_hoplite.db import brutalism
from perch_hoplite.db import datatypes
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import interface
from perch_hoplite.db import score_functions
from perch_hoplite.db import search_results
import pyarrow as pa
import pyarrow.parquet as pq

METADATA_FILENAME = 'metadata.parquet'
DEPLOYMENTS_FILENAME = 'deployments.parquet'
RECORDINGS_FILENAME = 'recordings.parquet'
WINDOWS_FILENAME = 'windows.parquet'
ANNOTATIONS_FILENAME = 'annotations.parquet'


@dataclasses.dataclass
class ParquetDB(interface.HopliteDBInterface):
  """Parquet hoplite database implementation."""

  db_path: epath.Path
  _embedding_dim: int
  _embedding_dtype: type[Any] = np.float16

  # In-memory pandas DataFrames
  _metadata: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
  _deployments: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
  _recordings: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
  _windows: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
  _annotations: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
  # Buffer for new embeddings not yet committed to parquet.
  _new_embeddings: pd.DataFrame = dataclasses.field(
      default_factory=pd.DataFrame
  )
  _id_counters: dict[str, int] = dataclasses.field(default_factory=dict)

  @property
  def metadata_path(self) -> epath.Path:
    return self.db_path / METADATA_FILENAME

  @property
  def deployments_path(self) -> epath.Path:
    return self.db_path / DEPLOYMENTS_FILENAME

  @property
  def recordings_path(self) -> epath.Path:
    return self.db_path / RECORDINGS_FILENAME

  @property
  def windows_path(self) -> epath.Path:
    return self.db_path / WINDOWS_FILENAME

  @property
  def annotations_path(self) -> epath.Path:
    return self.db_path / ANNOTATIONS_FILENAME

  @classmethod
  def create(
      cls,
      db_path: str,
      embedding_dim: int,
      embedding_dtype: type[Any] = np.float16,
  ) -> 'ParquetDB':
    """Connect to and, if needed, initialize the database."""
    db_path = epath.Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = db_path / METADATA_FILENAME
    if metadata_path.exists():
      metadata_df = pd.read_parquet(metadata_path)
    else:
      metadata_df = pd.DataFrame(columns=['key', 'value'])

    hoplite_db = cls(
        db_path=db_path,
        _embedding_dim=embedding_dim,
        _embedding_dtype=embedding_dtype,
        _metadata=metadata_df,
        _deployments=pd.DataFrame(),
        _recordings=pd.DataFrame(),
        _windows=pd.DataFrame(),
        _annotations=pd.DataFrame(),
        _id_counters={
            'deployments': 0,
            'recordings': 0,
            'windows': 0,
            'annotations': 0,
        },
    )
    hoplite_db._load_dataframes()
    hoplite_db.commit()
    return hoplite_db

  def _load_dataframes(self):
    """Load all dataframes from parquet files if they exist."""
    if self.deployments_path.exists():
      self._deployments = pd.read_parquet(self.deployments_path)
    else:
      self._deployments = pd.DataFrame(
          columns=['id', 'name', 'project', 'latitude', 'longitude']
      )

    if self.recordings_path.exists():
      self._recordings = pd.read_parquet(self.recordings_path)
    else:
      self._recordings = pd.DataFrame(
          columns=['id', 'filename', 'datetime', 'deployment_id']
      )

    if self.windows_path.exists():
      self._windows = pd.read_parquet(self.windows_path)
    else:
      self._windows = pd.DataFrame(columns=['id', 'recording_id', 'offsets'])

    if self.annotations_path.exists():
      self._annotations = pd.read_parquet(self.annotations_path)
    else:
      self._annotations = pd.DataFrame(
          columns=[
              'id',
              'recording_id',
              'offsets',
              'label',
              'label_type',
              'provenance',
          ]
      )

    self._id_counters['deployments'] = (
        self._deployments['id'].max() + 1 if not self._deployments.empty else 0
    )
    self._id_counters['recordings'] = (
        self._recordings['id'].max() + 1 if not self._recordings.empty else 0
    )
    self._id_counters['windows'] = (
        self._windows['id'].max() + 1 if not self._windows.empty else 0
    )
    self._id_counters['annotations'] = (
        self._annotations['id'].max() + 1 if not self._annotations.empty else 0
    )

  def _get_next_id(self, table_name: str) -> int:
    next_id = self._id_counters[table_name]
    self._id_counters[table_name] += 1
    return next_id

  def add_extra_table_column(
      self,
      table_name: str,
      column_name: str,
      column_type: type[Any],
  ) -> None:
    raise NotImplementedError()

  def get_extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    raise NotImplementedError()

  def commit(self) -> None:
    """Commit any pending transactions to the database."""
    self._metadata.to_parquet(self.metadata_path)
    self._deployments.to_parquet(self.deployments_path)
    self._recordings.to_parquet(self.recordings_path)
    self._windows.to_parquet(self.windows_path)
    if not self._new_embeddings.empty:
      t = pa.Table.from_pandas(self._new_embeddings)
      pq.write_to_dataset(
          t,
          self.db_path / 'embeddings',
          existing_data_behavior='overwrite_or_ignore',
      )
      self._new_embeddings = self._new_embeddings.iloc[0:0]
    self._annotations.to_parquet(self.annotations_path)

  def rollback(self) -> None:
    """Rollback any pending transactions to the database."""
    self._load_dataframes()

  def thread_split(self) -> 'ParquetDB':
    """Get a new instance of the Parquet DB."""
    return self.create(
        self.db_path.as_posix(),
        self._embedding_dim,
        self._embedding_dtype,
    )

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    """Insert a key-value pair into the metadata table."""
    json_coded = value.to_json()
    if key in self._metadata['key'].values:
      self._metadata.loc[self._metadata['key'] == key, 'value'] = json_coded
    else:
      new_row = pd.DataFrame({'key': [key], 'value': [json_coded]})
      self._metadata = pd.concat([self._metadata, new_row], ignore_index=True)

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    """Get a key-value pair from the metadata table."""
    if key is None:
      return config_dict.ConfigDict({
          r['key']: json.loads(r['value']) for _, r in self._metadata.iterrows()
      })

    result = self._metadata[self._metadata['key'] == key]
    if result.empty:
      raise KeyError(f'Metadata key not found: {key}')
    return config_dict.ConfigDict(json.loads(result['value'].iloc[0]))

  def remove_metadata(self, key: str | None) -> None:
    """Remove a key-value pair from the metadata table."""
    if key is None:
      self._metadata = self._metadata.iloc[0:0]
      return

    initial_len = len(self._metadata)
    self._metadata = self._metadata[self._metadata['key'] != key]
    if len(self._metadata) == initial_len:
      raise KeyError(f'Metadata key not found: {key}')

  def insert_deployment(
      self,
      name: str,
      project: str,
      latitude: float | None = None,
      longitude: float | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a deployment into the database."""
    for key in kwargs:
      if key not in self._deployments.columns:
        self._deployments[key] = None

    existing = self._deployments[
        (self._deployments['name'] == name)
        & (self._deployments['project'] == project)
    ]

    if not existing.empty:
      deployment_id = existing['id'].iloc[0]
      row_indexer = self._deployments['id'] == deployment_id
      self._deployments.loc[row_indexer, 'latitude'] = latitude
      self._deployments.loc[row_indexer, 'longitude'] = longitude
      for key, value in kwargs.items():
        self._deployments.loc[row_indexer, key] = value
      return deployment_id
    else:
      deployment_id = self._get_next_id('deployments')
      new_row_data = {
          'id': deployment_id,
          'name': name,
          'project': project,
          'latitude': latitude,
          'longitude': longitude,
          **kwargs,
      }
      new_row = pd.DataFrame([new_row_data])
      self._deployments = pd.concat(
          [self._deployments, new_row], ignore_index=True
      )
      return deployment_id

  def get_deployment(self, deployment_id: int) -> datatypes.Deployment:
    """Get a deployment from the database."""
    result = self._deployments[self._deployments['id'] == deployment_id]
    if result.empty:
      raise KeyError(f'Deployment id not found: {deployment_id}')
    return datatypes.Deployment(**result.iloc[0].to_dict())

  def remove_deployment(self, deployment_id: int) -> None:
    self._deployments = self._deployments[
        self._deployments['id'] != deployment_id
    ]

  def insert_recording(
      self,
      filename: str,
      datetime: dt.datetime | None = None,
      deployment_id: int | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a recording into the database."""
    for key in kwargs:
      if key not in self._recordings.columns:
        self._recordings[key] = None

    if deployment_id is None:
      existing = self._recordings[
          (self._recordings['filename'] == filename)
          & self._recordings['deployment_id'].isna()
      ]
    else:
      existing = self._recordings[
          (self._recordings['filename'] == filename)
          & (self._recordings['deployment_id'] == deployment_id)
      ]

    if not existing.empty:
      recording_id = existing['id'].iloc[0]
      row_indexer = self._recordings['id'] == recording_id
      self._recordings.loc[row_indexer, 'datetime'] = datetime
      for key, value in kwargs.items():
        self._recordings.loc[row_indexer, key] = value
      return recording_id
    else:
      recording_id = self._get_next_id('recordings')
      new_row_data = {
          'id': recording_id,
          'filename': filename,
          'datetime': datetime,
          'deployment_id': deployment_id,
          **kwargs,
      }
      new_row = pd.DataFrame([new_row_data])
      self._recordings = pd.concat(
          [self._recordings, new_row], ignore_index=True
      )
      return recording_id

  def get_recording(self, recording_id: int) -> datatypes.Recording:
    """Get a recording from the database."""
    result = self._recordings[self._recordings['id'] == recording_id]
    if result.empty:
      raise KeyError(f'Recording id not found: {recording_id}')
    return datatypes.Recording(**result.iloc[0].to_dict())

  def remove_recording(self, recording_id: int) -> None:
    self._recordings = self._recordings[self._recordings['id'] != recording_id]

  def insert_window(
      self,
      recording_id: int,
      offsets: list[float],
      embedding: np.ndarray | None = None,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
      **kwargs: Any,
  ) -> int:
    """Insert a window into the database."""
    duplicate_id = self._handle_window_duplicates(
        recording_id, offsets, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

    if embedding is not None and embedding.shape[-1] != self._embedding_dim:
      raise ValueError(
          f'Incorrect embedding dimension. Expected {self._embedding_dim}, but'
          f' got {embedding.shape[-1]}.'
      )

    for key in kwargs:
      if key not in self._windows.columns:
        self._windows[key] = None

    window_id = self._get_next_id('windows')
    new_row_data = {
        'id': window_id,
        'recording_id': recording_id,
        'offsets': offsets,
        **kwargs,
    }
    new_row = pd.DataFrame([new_row_data])
    self._windows = pd.concat([self._windows, new_row], ignore_index=True)
    if embedding is not None:
      new_embedding = pd.DataFrame({
          'id': [window_id],
          'embedding': [embedding.tolist()],
      })
      self._new_embeddings = pd.concat(
          [self._new_embeddings, new_embedding], ignore_index=True
      )
    return window_id

  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> datatypes.Window:
    """Get a window from the database."""
    result = self._windows[self._windows['id'] == window_id]
    if result.empty:
      raise KeyError(f'Window id not found: {window_id}')

    window_data = result.iloc[0].to_dict()
    window = datatypes.Window(embedding=None, **window_data)
    if include_embedding:
      window.embedding = self.get_embedding(window_id)
    return window

  def get_embedding(self, window_id: int) -> np.ndarray:
    """Get an embedding vector from the database."""
    window_id = int(window_id)
    if not self._new_embeddings.empty:
      result = self._new_embeddings[self._new_embeddings['id'] == window_id]
      if not result.empty:
        embedding = result.iloc[0]['embedding']
        return np.array(embedding, dtype=self._embedding_dtype)
    if (self.db_path / 'embeddings').exists():
      try:
        df = pq.read_table(
            self.db_path / 'embeddings', filters=[('id', '==', window_id)]
        ).to_pandas()
        if not df.empty:
          embedding = df.iloc[0]['embedding']
          return np.array(embedding, dtype=self._embedding_dtype)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            'Could not read from embeddings dataset %s : %s',
            self.db_path / 'embeddings',
            e,
        )
    raise KeyError(f'Embedding vector not found for window id: {window_id}')

  def get_embeddings_batch(self, window_ids: Sequence[int]) -> np.ndarray:
    """Get a batch of embedding vectors from the database."""
    embeddings = {}
    if not self._new_embeddings.empty:
      results = self._new_embeddings[
          self._new_embeddings['id'].isin(window_ids)
      ]
      for _, row in results.iterrows():
        embeddings[row['id']] = row['embedding']

    remaining_ids = [i for i in window_ids if i not in embeddings]
    if remaining_ids and (self.db_path / 'embeddings').exists():
      try:
        df = pq.read_table(
            self.db_path / 'embeddings', filters=[('id', 'in', remaining_ids)]
        ).to_pandas()
        for _, row in df.iterrows():
          embeddings[row['id']] = row['embedding']
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            'Could not read from embeddings dataset %s : %s',
            self.db_path / 'embeddings',
            e,
        )

    output = []
    for i in window_ids:
      if i not in embeddings:
        raise KeyError(f'Embedding vector not found for window id: {i}')
      output.append(embeddings[i])
    return np.array(output, dtype=self._embedding_dtype)

  def remove_window(self, window_id: int) -> None:
    self._windows = self._windows[self._windows['id'] != window_id]

  def insert_annotation(
      self,
      recording_id: int,
      offsets: list[float],
      label: str,
      label_type: datatypes.LabelType,
      provenance: str,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
      **kwargs: Any,
  ) -> int:
    """Insert an annotation into the database."""
    duplicate_id = self._handle_annotation_duplicates(
        recording_id, offsets, label, label_type, provenance, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

    for key in kwargs:
      if key not in self._annotations.columns:
        self._annotations[key] = None

    annotation_id = self._get_next_id('annotations')
    new_row_data = {
        'id': annotation_id,
        'recording_id': recording_id,
        'offsets': offsets,
        'label': label,
        'label_type': label_type.value,
        'provenance': provenance,
        **kwargs,
    }
    new_row = pd.DataFrame([new_row_data])
    self._annotations = pd.concat(
        [self._annotations, new_row], ignore_index=True
    )
    return annotation_id

  def get_annotation(self, annotation_id: int) -> datatypes.Annotation:
    """Get an annotation from the database."""
    result = self._annotations[self._annotations['id'] == annotation_id]
    if result.empty:
      raise KeyError(f'Annotation id not found: {annotation_id}')

    annotation_data = result.iloc[0].to_dict()
    annotation_data['label_type'] = datatypes.LabelType(
        annotation_data['label_type']
    )
    return datatypes.Annotation(**annotation_data)

  def remove_annotation(self, annotation_id: int) -> None:
    self._annotations = self._annotations[
        self._annotations['id'] != annotation_id
    ]

  def count_embeddings(self) -> int:
    """Get the number of embeddings in the database."""
    return len(self._windows)

  def match_window_ids(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      annotations_filter: config_dict.ConfigDict | None = None,
      limit: int | None = None,
  ) -> Sequence[int]:
    """Get matching window IDs from the database based on given filters."""

    if limit is not None and limit <= 0:
      raise ValueError('Limit must be None or positive.')

    # Filter by deployment constraints.
    if deployments_filter:
      deployments = self.get_all_deployments(filter=deployments_filter)
      restrict_deployments = {d.id for d in deployments}
    else:
      restrict_deployments = None

    # Filter by recording constraints.
    if recordings_filter or restrict_deployments is not None:
      recordings = self.get_all_recordings(filter=recordings_filter)
      restrict_recordings = {r.id for r in recordings}
      if restrict_deployments is not None:
        restrict_recordings &= {
            r.id for r in recordings if r.deployment_id in restrict_deployments
        }
    else:
      restrict_recordings = None

    # Filter by window constraints.
    if windows_filter or restrict_recordings is not None:
      windows = self.get_all_windows(filter=windows_filter)
      restrict_windows = {w.id for w in windows}
      if restrict_recordings is not None:
        restrict_windows &= {
            w.id for w in windows if w.recording_id in restrict_recordings
        }
    else:
      restrict_windows = set(self._windows['id'].tolist())

    # Filter by annotation constraints.
    if annotations_filter:
      annotations = self.get_all_annotations(filter=annotations_filter)
      restrict_annotations = {a.id for a in annotations}
      restrict_recording_offsets = {
          (
              a.recording_id,
              tuple(a.offsets),
          )
          for a in annotations
          if a.id in restrict_annotations
      }

      window_objects = self.get_all_windows(filter=windows_filter)
      restrict_windows &= {
          w.id
          for w in window_objects
          if (w.recording_id, tuple(w.offsets)) in restrict_recording_offsets
      }

    # Return the window IDs that match the constraints.
    if limit is None:
      return list(restrict_windows)
    return list(itertools.islice(restrict_windows, limit))

  def get_all_projects(self) -> Sequence[str]:
    """Get all distinct projects from the database."""
    return sorted(self._deployments['project'].unique().tolist())

  def get_all_deployments(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Deployment]:
    """Get all deployments from the database."""
    if not filter:
      return [
          datatypes.Deployment(**row.to_dict())
          for _, row in self._deployments.iterrows()
      ]
    deployment_objects = {
        r['id']: datatypes.Deployment(**r)
        for r in self._deployments.to_dict('records')
    }
    keys = in_mem_impl.select_matching_keys(deployment_objects, filter)
    return [deployment_objects[k] for k in keys]

  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Recording]:
    """Get all recordings from the database."""
    if not filter:
      return [
          datatypes.Recording(**row.to_dict())
          for _, row in self._recordings.iterrows()
      ]
    recording_objects = {
        r['id']: datatypes.Recording(**r)
        for r in self._recordings.to_dict('records')
    }
    keys = in_mem_impl.select_matching_keys(recording_objects, filter)
    return [recording_objects[k] for k in keys]

  def get_all_windows(
      self,
      include_embedding: bool = False,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Window]:
    """Get all windows from the database."""
    df = self._windows
    if filter:
      window_objects = {
          r['id']: datatypes.Window(**r, embedding=None)
          for r in self._windows.to_dict('records')
      }
      keys = in_mem_impl.select_matching_keys(window_objects, filter)
      df = self._windows[self._windows['id'].isin(keys)]

    windows = []
    for _, row in df.iterrows():
      window_data = row.to_dict()
      window = datatypes.Window(embedding=None, **window_data)
      if include_embedding:
        window.embedding = self.get_embedding(window.id)
      else:
        window.embedding = None
      windows.append(window)
    return windows

  def get_all_annotations(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Annotation]:
    """Get all annotations from the database."""
    df = self._annotations
    if filter:
      annotation_objects = {
          r['id']: datatypes.Annotation(
              **r | {'label_type': datatypes.LabelType(r['label_type'])}
          )
          for r in self._annotations.to_dict('records')
      }
      keys = in_mem_impl.select_matching_keys(annotation_objects, filter)
      df = self._annotations[self._annotations['id'].isin(keys)]

    annotations = []
    for _, row in df.iterrows():
      annotation_data = row.to_dict()
      annotation_data['label_type'] = datatypes.LabelType(
          annotation_data['label_type']
      )
      annotations.append(datatypes.Annotation(**annotation_data))
    return annotations

  def get_all_labels(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> Sequence[str]:
    """Get all distinct labels from the database."""
    df = self._annotations
    if label_type is not None:
      df = df[df['label_type'] == label_type.value]
    return sorted(df['label'].unique())

  def count_each_label(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> collections.Counter[str]:
    """Count each label in the database, ignoring provenance."""
    df = self._annotations
    if label_type is not None:
      df = df[df['label_type'] == label_type.value]
    df['offsets'] = df['offsets'].apply(tuple)
    df = df[
        ['recording_id', 'offsets', 'label', 'label_type']
    ].drop_duplicates()
    return collections.Counter(df['label'].tolist())

  def get_embedding_dim(self) -> int:
    return self._embedding_dim

  def get_embedding_dtype(self) -> type[Any]:
    return self._embedding_dtype

  def search(
      self,
      query_embedding: np.ndarray,
      search_list_size: int,
      approximate: bool = True,
      target_score: float | None = None,
      score_fn_name: str = 'dot',
      **kwargs: Any,
  ) -> search_results.TopKSearchResults:
    """Search for neighbors of the query embedding."""
    if approximate:
      raise NotImplementedError('Approximate search is not supported.')
    score_fn = score_functions.get_score_fn(
        score_fn_name, target_score=target_score
    )
    results = brutalism.threaded_brute_search(
        self,
        query_embedding,
        search_list_size,
        score_fn=score_fn,
        **kwargs,
    )

    if target_score is not None:
      raw_score_fn = score_functions.get_score_fn(
          score_fn_name, target_score=None
      )
      # If target_sampling, the results should be returned with the original
      # (un-targeted) score.
      results = brutalism.rerank(query_embedding, results, self, raw_score_fn)
    return results
