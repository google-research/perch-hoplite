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

"""Implementation of a MultiDB wrapper for HopliteDBInterface."""

import collections
from collections.abc import Sequence
import datetime as dt
from typing import Any, Literal

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import interface


class MultiDBWrapper(interface.HopliteDBInterface):
  """A wrapper around multiple HopliteDBInterface instances.

  ID numbers are translated internally to avoid collisions.
  """

  def __init__(self, dbs: Sequence[interface.HopliteDBInterface]):
    """Initialize with multiple databases.

    Args:
      dbs: A sequence of databases to wrap.

    Raises:
      ValueError: If `MultiDBWrapper` contains no databases.
    """
    self._dbs = dbs
    if not self._dbs:
      raise ValueError("MultiDBWrapper contains no databases.")

  def _split_id(self, external_id: int) -> tuple[int, int]:
    """Splits an external ID into a database index and internal ID.

    Calculate the internal database index and internal ID from an external ID
    number.

    Args:
      external_id: The external ID number.

    Returns:
      db_index: The database index.
      internal_id: The internal ID number within the database that has index
        db_index.
    """

    num_dbs = len(self._dbs)
    db_index = external_id % num_dbs
    internal_id = external_id // num_dbs
    self._dbs[db_index].get_window(internal_id)
    return db_index, internal_id

  def _join_id(self, db_index: int, internal_id: int) -> int:
    """Joins a database index and internal ID into an external ID.

    Args:
      db_index: The database index.
      internal_id: The internal ID number.

    Returns:
      The external ID number.
    """

    self._dbs[db_index].get_window(internal_id)
    return internal_id * len(self._dbs) + db_index

  @classmethod
  def create(cls, **kwargs) -> "MultiDBWrapper":
    raise NotImplementedError()

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
    for db in self._dbs:
      db.commit()

  def rollback(self) -> None:
    for db in self._dbs:
      db.rollback()

  def thread_split(self) -> "MultiDBWrapper":
    return MultiDBWrapper([db.thread_split() for db in self._dbs])

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    raise NotImplementedError()

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    raise NotImplementedError()

  def remove_metadata(self, key: str | None) -> None:
    raise NotImplementedError()

  def insert_deployment(
      self,
      name: str,
      project: str,
      latitude: float | None = None,
      longitude: float | None = None,
      **kwargs: Any,
  ) -> int:
    raise NotImplementedError()

  def get_deployment(self, deployment_id: int) -> datatypes.Deployment:
    db_index, internal_id = self._split_id(deployment_id)
    deployment = self._dbs[db_index].get_deployment(internal_id)
    # The returned Deployment is the same as the internal Deployment, except
    # its ID is the external deployment_id argument of get_deployment().
    return datatypes.Deployment(
        **deployment.to_kwargs(skip=["id"]), id=deployment_id
    )

  def remove_deployment(self, deployment_id: int) -> None:
    db_index, internal_id = self._split_id(deployment_id)
    self._dbs[db_index].remove_deployment(internal_id)

  def insert_recording(
      self,
      filename: str,
      datetime: dt.datetime | None = None,
      deployment_id: int | None = None,
      **kwargs: Any,
  ) -> int:
    if deployment_id is None:
      raise NotImplementedError(
          "MultiDBWrapper requires deployment_id to insert a recording."
      )
    db_index, internal_dep_id = self._split_id(deployment_id)
    internal_rec_id = self._dbs[db_index].insert_recording(
        filename=filename,
        datetime=datetime,
        deployment_id=internal_dep_id,
        **kwargs,
    )
    return self._join_id(db_index, internal_rec_id)

  def get_recording(self, recording_id: int) -> datatypes.Recording:
    db_index, internal_rec_id = self._split_id(recording_id)
    recording = self._dbs[db_index].get_recording(internal_rec_id)
    deployment_id = None
    if recording.deployment_id is not None:
      deployment_id = self._join_id(db_index, recording.deployment_id)
    return datatypes.Recording(
        **recording.to_kwargs(skip=["id", "deployment_id"]),
        id=recording_id,
        deployment_id=deployment_id,
    )

  def remove_recording(self, recording_id: int) -> None:
    db_index, internal_id = self._split_id(recording_id)
    self._dbs[db_index].remove_recording(internal_id)

  def insert_window(
      self,
      recording_id: int,
      offsets: list[float],
      embedding: np.ndarray | None = None,
      handle_duplicates: Literal[
          "allow", "overwrite", "skip", "error"
      ] = "error",
      **kwargs: Any,
  ) -> int:
    db_index, internal_rec_id = self._split_id(recording_id)
    internal_win_id = self._dbs[db_index].insert_window(
        recording_id=internal_rec_id,
        offsets=offsets,
        embedding=embedding,
        handle_duplicates=handle_duplicates,
        **kwargs,
    )
    return self._join_id(db_index, internal_win_id)

  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> datatypes.Window:
    db_index, internal_win_id = self._split_id(window_id)
    window = self._dbs[db_index].get_window(internal_win_id, include_embedding)
    return datatypes.Window(
        **window.to_kwargs(skip=["id", "recording_id"]),
        id=window_id,
        recording_id=self._join_id(db_index, window.recording_id),
    )

  def get_embedding(self, window_id: int) -> np.ndarray:
    db_index, internal_win_id = self._split_id(window_id)
    return self._dbs[db_index].get_embedding(internal_win_id)

  def remove_window(self, window_id: int) -> None:
    db_index, internal_win_id = self._split_id(window_id)
    self._dbs[db_index].remove_window(internal_win_id)

  def insert_annotation(
      self,
      recording_id: int,
      offsets: list[float],
      label: str,
      label_type: datatypes.LabelType,
      provenance: str,
      handle_duplicates: Literal[
          "allow", "overwrite", "skip", "error"
      ] = "error",
      **kwargs: Any,
  ) -> int:
    db_index, internal_rec_id = self._split_id(recording_id)
    internal_ann_id = self._dbs[db_index].insert_annotation(
        recording_id=internal_rec_id,
        offsets=offsets,
        label=label,
        label_type=label_type,
        provenance=provenance,
        handle_duplicates=handle_duplicates,
        **kwargs,
    )
    return self._join_id(db_index, internal_ann_id)

  def get_annotation(self, annotation_id: int) -> datatypes.Annotation:
    db_index, internal_ann_id = self._split_id(annotation_id)
    annotation = self._dbs[db_index].get_annotation(internal_ann_id)
    return datatypes.Annotation(
        **annotation.to_kwargs(skip=["id", "recording_id"]),
        id=annotation_id,
        recording_id=self._join_id(db_index, annotation.recording_id),
    )

  def remove_annotation(self, annotation_id: int) -> None:
    db_index, internal_ann_id = self._split_id(annotation_id)
    self._dbs[db_index].remove_annotation(internal_ann_id)

  def count_embeddings(self) -> int:
    return sum(db.count_embeddings() for db in self._dbs)

  def match_window_ids(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      annotations_filter: config_dict.ConfigDict | None = None,
      limit: int | None = None,
  ) -> Sequence[int]:
    raise NotImplementedError()

  def get_all_projects(self) -> Sequence[str]:
    projects = set()
    for db in self._dbs:
      projects.update(db.get_all_projects())
    return sorted(list(projects))

  def get_all_deployments(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Deployment]:
    raise NotImplementedError()

  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Recording]:
    raise NotImplementedError()

  def get_all_windows(
      self,
      include_embedding: bool = False,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Window]:
    raise NotImplementedError()

  def get_all_annotations(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Annotation]:
    raise NotImplementedError()

  def get_all_labels(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> Sequence[str]:
    labels = set()
    for db in self._dbs:
      labels.update(db.get_all_labels(label_type=label_type))
    return sorted(list(labels))

  def count_each_label(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> collections.Counter[str]:
    counts = collections.Counter()
    for db in self._dbs:
      counts.update(db.count_each_label(label_type=label_type))
    return counts

  def get_embedding_dim(self) -> int:
    return self._dbs[0].get_embedding_dim()

  def get_embedding_dtype(self) -> type[Any]:
    return self._dbs[0].get_embedding_dtype()
