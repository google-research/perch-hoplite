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

"""In-memory database implementation."""

import collections
from collections.abc import Sequence
import copy
import dataclasses
import datetime as dt
import itertools
from typing import Any, Literal
from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import filter_utils
from perch_hoplite.db import interface


@dataclasses.dataclass
class InMemoryGraphSearchDB(interface.HopliteDBInterface):
  """In-memory hoplite database."""

  # User-provided.
  _embedding_dim: int
  _embedding_dtype: type[Any]

  # Dynamic state.
  _extra_table_columns: dict[str, dict[str, type[Any]]] = dataclasses.field(
      default_factory=lambda: {
          'deployments': {},
          'recordings': {},
          'windows': {},
          'annotations': {},
      }
  )

  # Storage for metadata & embeddings.
  _hoplite_metadata: dict[str, config_dict.ConfigDict] = dataclasses.field(
      default_factory=dict
  )
  _deployments: dict[int, datatypes.Deployment] = dataclasses.field(
      default_factory=dict
  )
  _recordings: dict[int, datatypes.Recording] = dataclasses.field(
      default_factory=dict
  )
  _windows: dict[int, datatypes.Window] = dataclasses.field(
      default_factory=dict
  )
  _annotations: dict[int, datatypes.Annotation] = dataclasses.field(
      default_factory=dict
  )
  _next_deployment_id: int = 1
  _next_recording_id: int = 1
  _next_window_id: int = 1
  _next_annotation_id: int = 1

  @classmethod
  def create(
      cls,
      embedding_dim: int,
      embedding_dtype: type[Any] = np.float16,
  ) -> 'InMemoryGraphSearchDB':
    """Connect to and, if needed, initialize the database."""
    return cls(
        _embedding_dim=embedding_dim,
        _embedding_dtype=embedding_dtype,
    )

  def add_extra_table_column(
      self,
      table_name: str,
      column_name: str,
      column_type: type[Any],
  ) -> None:
    """Add an extra column to a table in the database.."""
    self._extra_table_columns[table_name][column_name] = column_type

  def get_extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    """Get all extra columns in the database."""
    return self._extra_table_columns

  def commit(self) -> None:
    """No-op to commit any pending transactions to the database.."""
    pass

  def rollback(self) -> None:
    """No-op to rollback any pending transactions to the database."""
    pass

  def thread_split(self) -> 'InMemoryGraphSearchDB':
    """Return the same database object since all data is in shared memory."""
    return self

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    """Insert a key-value pair into the metadata table."""
    self._hoplite_metadata[key] = value

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    """Get a key-value pair from the metadata table."""

    if key is None:
      return config_dict.ConfigDict(self._hoplite_metadata)
    else:
      return self._hoplite_metadata[key]

  def remove_metadata(self, key: str | None) -> None:
    """Remove a key-value pair from the metadata table."""

    if key is None:
      self._hoplite_metadata.clear()
    else:
      del self._hoplite_metadata[key]

  def insert_deployment(
      self,
      name: str,
      project: str,
      latitude: float | None = None,
      longitude: float | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a deployment into the database."""

    deployment_id = self._next_deployment_id
    self._deployments[deployment_id] = datatypes.Deployment(
        id=deployment_id,
        name=name,
        project=project,
        latitude=latitude,
        longitude=longitude,
        **kwargs,
    )
    self._next_deployment_id += 1
    return deployment_id

  def get_deployment(self, deployment_id: int) -> datatypes.Deployment:
    """Get a deployment from the database."""
    deployment_id = int(deployment_id)
    return self._deployments[deployment_id]

  def remove_deployment(self, deployment_id: int) -> None:
    """Remove a deployment from the database."""

    deployment_id = int(deployment_id)

    remove_recording_ids = [
        recording.id
        for recording in self._recordings.values()
        if recording.deployment_id == deployment_id
    ]
    for recording_id in remove_recording_ids:
      self.remove_recording(recording_id)
    del self._deployments[deployment_id]

  def insert_recording(
      self,
      filename: str,
      datetime: dt.datetime | None = None,
      deployment_id: int | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a recording into the database."""

    if deployment_id is not None and deployment_id not in self._deployments:
      raise ValueError(f'Deployment id not found: {deployment_id}')

    recording_id = self._next_recording_id
    self._recordings[recording_id] = datatypes.Recording(
        id=recording_id,
        filename=filename,
        datetime=datetime,
        deployment_id=deployment_id,
        **kwargs,
    )
    self._next_recording_id += 1
    return recording_id

  def get_recording(
      self,
      recording_id: int,
  ) -> datatypes.Recording:
    """Get a recording from the database."""
    recording_id = int(recording_id)
    return self._recordings[recording_id]

  def remove_recording(self, recording_id: int) -> None:
    """Remove a recording from the database."""

    recording_id = int(recording_id)

    remove_window_ids = [
        window.id
        for window in self._windows.values()
        if window.recording_id == recording_id
    ]
    for window_id in remove_window_ids:
      self.remove_window(window_id)

    remove_annotation_ids = [
        annotation.id
        for annotation in self._annotations.values()
        if annotation.recording_id == recording_id
    ]
    for annotation_id in remove_annotation_ids:
      self.remove_annotation(annotation_id)

    del self._recordings[recording_id]

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

    if recording_id not in self._recordings:
      raise ValueError(f'Recording id not found: {recording_id}')

    duplicate_id = self._handle_window_duplicates(
        recording_id, offsets, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

    window_id = self._next_window_id
    self._windows[window_id] = datatypes.Window(
        id=window_id,
        recording_id=recording_id,
        offsets=offsets,
        embedding=embedding,
        **kwargs,
    )
    self._next_window_id += 1
    return window_id

  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> datatypes.Window:
    """Get a window from the database."""

    window_id = int(window_id)
    window = self._windows[window_id]

    if include_embedding:
      return window

    window = copy.copy(window)
    window.embedding = None
    return window

  def get_embedding(self, window_id: int) -> np.ndarray:
    """Get an embedding vector from the database."""
    window_id = int(window_id)
    return self._windows[window_id].embedding

  def remove_window(self, window_id: int) -> None:
    """Remove a window from the database."""
    window_id = int(window_id)
    del self._windows[window_id]

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

    if recording_id not in self._recordings:
      raise ValueError(f'Recording id not found: {recording_id}')

    duplicate_id = self._handle_annotation_duplicates(
        recording_id, offsets, label, label_type, provenance, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

    annotation_id = self._next_annotation_id
    self._annotations[annotation_id] = datatypes.Annotation(
        id=annotation_id,
        recording_id=recording_id,
        offsets=offsets,
        label=label,
        label_type=label_type,
        provenance=provenance,
        **kwargs,
    )
    self._next_annotation_id += 1
    return annotation_id

  def get_annotation(self, annotation_id: int) -> datatypes.Annotation:
    """Get an annotation from the database."""
    annotation_id = int(annotation_id)
    return self._annotations[annotation_id]

  def remove_annotation(self, annotation_id: int) -> None:
    """Remove an annotation from the database."""
    annotation_id = int(annotation_id)
    del self._annotations[annotation_id]

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
      restrict_deployments = filter_utils.select_matching_keys(
          self._deployments,
          deployments_filter,
      )
    else:
      restrict_deployments = None

    # Filter by recording constraints.
    if recordings_filter or restrict_deployments is not None:
      if recordings_filter:
        restrict_recordings = filter_utils.select_matching_keys(
            self._recordings,
            recordings_filter,
        )
      else:
        restrict_recordings = set(self._recordings.keys())
      if restrict_deployments is not None:
        restrict_recordings &= {
            key
            for key, value in self._recordings.items()
            if value.deployment_id in restrict_deployments
        }
    else:
      restrict_recordings = None

    # Filter by window constraints.
    if windows_filter or restrict_recordings is not None:
      if windows_filter:
        restrict_windows = filter_utils.select_matching_keys(
            self._windows,
            windows_filter,
        )
      else:
        restrict_windows = set(self._windows.keys())
      if restrict_recordings is not None:
        restrict_windows &= {
            key
            for key, value in self._windows.items()
            if value.recording_id in restrict_recordings
        }
    else:
      restrict_windows = set(self._windows.keys())

    # Filter by annotation constraints.
    if annotations_filter:
      restrict_annotations = filter_utils.select_matching_keys(
          self._annotations,
          annotations_filter,
      )
      restrict_recording_offsets = {
          (
              self._annotations[annotation_id].recording_id,
              tuple(self._annotations[annotation_id].offsets),
          )
          for annotation_id in restrict_annotations
      }
      restrict_windows = {
          window_id
          for window_id in restrict_windows
          if (
              self._windows[window_id].recording_id,
              tuple(self._windows[window_id].offsets),
          )
          in restrict_recording_offsets
      }

    # Return the window IDs that match the constraints.
    if limit is None:
      return list(restrict_windows)
    return list(itertools.islice(restrict_windows, limit))

  def get_all_projects(self) -> Sequence[str]:
    """Get all distinct projects from the database."""
    return sorted({d.project for d in self._deployments.values()})

  def get_all_deployments(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Deployment]:
    """Get all deployments from the database."""
    restrict_deployments = filter_utils.select_matching_keys(
        self._deployments, filter
    )
    return [self._deployments[key] for key in restrict_deployments]

  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Recording]:
    """Get all recordings from the database."""
    restrict_recordings = filter_utils.select_matching_keys(
        self._recordings, filter
    )
    return [self._recordings[key] for key in restrict_recordings]

  def get_all_windows(
      self,
      include_embedding: bool = False,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
      annotations_filter: config_dict.ConfigDict | None = None,
  ) -> Sequence[datatypes.Window]:
    """Get all windows from the database."""
    window_ids = set(self._windows.keys())
    if filter:
      window_ids &= filter_utils.select_matching_keys(self._windows, filter)

    if deployments_filter:
      depl_ids = filter_utils.select_matching_keys(
          self._deployments, deployments_filter
      )
      rec_ids = {
          r_id
          for r_id, r in self._recordings.items()
          if r.deployment_id in depl_ids
      }
      window_ids &= {
          w_id
          for w_id in window_ids
          if self._windows[w_id].recording_id in rec_ids
      }

    if recordings_filter:
      rec_ids = filter_utils.select_matching_keys(
          self._recordings, recordings_filter
      )
      window_ids &= {
          w_id
          for w_id in window_ids
          if self._windows[w_id].recording_id in rec_ids
      }

    if annotations_filter:
      annot_ids = filter_utils.select_matching_keys(
          self._annotations, annotations_filter
      )
      ann_rec_offsets = {
          (
              self._annotations[annotation_id].recording_id,
              tuple(self._annotations[annotation_id].offsets),
          )
          for annotation_id in annot_ids
      }
      window_ids &= {
          w_id
          for w_id in window_ids
          if (
              self._windows[w_id].recording_id,
              tuple(self._windows[w_id].offsets),
          )
          in ann_rec_offsets
      }

    windows = [self._windows[key] for key in window_ids]

    if include_embedding:
      return windows

    windows = [copy.copy(window) for window in windows]
    for window in windows:
      window.embedding = None
    return windows

  def get_all_annotations(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Annotation]:
    """Get all annotations from the database."""
    restrict_annotations = filter_utils.select_matching_keys(
        self._annotations, filter
    )
    return [self._annotations[key] for key in restrict_annotations]

  def get_all_labels(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> Sequence[str]:
    """Get all distinct labels from the database."""
    return sorted({
        a.label
        for a in self._annotations.values()
        if label_type is None or a.label_type == label_type
    })

  def count_each_label(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> collections.Counter[str]:
    """Count each label in the database, ignoring provenance."""

    # Avoid double-counting the same label on the same recording offsets because
    # of different provenances.
    unique_annotations = {
        (a.recording_id, tuple(a.offsets), a.label, a.label_type)
        for a in self._annotations.values()
        if label_type is None or a.label_type == label_type
    }
    return collections.Counter([a[2] for a in unique_annotations])

  def get_embedding_dim(self) -> int:
    """Get the embedding dimension."""
    return self._embedding_dim

  def get_embedding_dtype(self) -> type[Any]:
    """Get the embedding data type."""
    return self._embedding_dtype
