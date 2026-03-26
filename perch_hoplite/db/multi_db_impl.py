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
    raise NotImplementedError()

  def rollback(self) -> None:
    raise NotImplementedError()

  def thread_split(self) -> "MultiDBWrapper":
    raise NotImplementedError()

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
    raise NotImplementedError()

  def remove_deployment(self, deployment_id: int) -> None:
    raise NotImplementedError()

  def insert_recording(
      self,
      filename: str,
      datetime: dt.datetime | None = None,
      deployment_id: int | None = None,
      **kwargs: Any,
  ) -> int:
    raise NotImplementedError()

  def get_recording(self, recording_id: int) -> datatypes.Recording:
    raise NotImplementedError()

  def remove_recording(self, recording_id: int) -> None:
    raise NotImplementedError()

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
    raise NotImplementedError()

  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> datatypes.Window:
    raise NotImplementedError()

  def get_embedding(self, window_id: int) -> np.ndarray:
    raise NotImplementedError()

  def remove_window(self, window_id: int) -> None:
    raise NotImplementedError()

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
    raise NotImplementedError()

  def get_annotation(self, annotation_id: int) -> datatypes.Annotation:
    raise NotImplementedError()

  def remove_annotation(self, annotation_id: int) -> None:
    raise NotImplementedError()

  def count_embeddings(self) -> int:
    raise NotImplementedError()

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
    raise NotImplementedError()

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
    raise NotImplementedError()

  def count_each_label(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> collections.Counter[str]:
    raise NotImplementedError()

  def get_embedding_dim(self) -> int:
    raise NotImplementedError()

  def get_embedding_dtype(self) -> type[Any]:
    raise NotImplementedError()
