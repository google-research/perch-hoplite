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
from collections.abc import Mapping, Sequence
import datetime as dt
from typing import Any, Literal

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import filter_utils
from perch_hoplite.db import interface


class MultiDBWrapper(interface.HopliteDBInterface):
  """A wrapper around multiple HopliteDBInterface instances.

  ID numbers are translated internally to avoid collisions.
  """

  def __init__(self, dbs: Mapping[str, interface.HopliteDBInterface]):
    """Initialize with multiple databases.

    Args:
      dbs: A mapping from database names to databases to wrap.

    Raises:
      ValueError: If `MultiDBWrapper` contains no databases.
    """
    self._dbs = dict(sorted(dbs.items()))
    if not self._dbs:
      raise ValueError("MultiDBWrapper contains no databases.")
    self._dbs_list = list(self._dbs.values())

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

    num_dbs = len(self._dbs_list)
    db_index = external_id % num_dbs
    internal_id = external_id // num_dbs
    return db_index, internal_id

  def _join_id(self, db_index: int, internal_id: int) -> int:
    """Joins a database index and internal ID into an external ID.

    Args:
      db_index: The database index.
      internal_id: The internal ID number.

    Returns:
      The external ID number.
    """

    return internal_id * len(self._dbs_list) + db_index

  @classmethod
  def create(
      cls,
      dbs: Mapping[str, interface.HopliteDBInterface],
  ) -> "MultiDBWrapper":
    return cls(dbs=dbs)

  def add_extra_table_column(
      self,
      table_name: str,
      column_name: str,
      column_type: type[Any],
  ) -> None:
    for db in self._dbs_list:
      db.add_extra_table_column(table_name, column_name, column_type)

  def get_extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    all_extra_table_columns = [
        db.get_extra_table_columns() for db in self._dbs_list
    ]
    table_names = all_extra_table_columns[0].keys()
    result = {}
    for table_name in table_names:
      extra_table_columns_sets = []
      for extra_table_columns in all_extra_table_columns:
        extra_table_columns_sets.append(
            set(extra_table_columns.get(table_name, {}).items())
        )
      common_extra_table_columns = set.intersection(*extra_table_columns_sets)
      result[table_name] = dict(common_extra_table_columns)
    return result

  def commit(self) -> None:
    for db in self._dbs_list:
      db.commit()

  def rollback(self) -> None:
    for db in self._dbs_list:
      db.rollback()

  def thread_split(self) -> "MultiDBWrapper":
    return MultiDBWrapper(
        {name: db.thread_split() for name, db in self._dbs.items()}
    )

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    if "/" not in key:
      raise ValueError(
          f"Metadata key {key} must be in format 'db_name/internal_key'"
      )
    db_name, internal_key = key.split("/", 1)
    if db_name not in self._dbs:
      raise ValueError(f"Database {db_name} not found in MultiDBWrapper")
    self._dbs[db_name].insert_metadata(internal_key, value)

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    if key is None:
      all_metadata = config_dict.ConfigDict()
      for db_name, db in self._dbs.items():
        db_metadata = db.get_metadata(None)
        for internal_key, value in db_metadata.items():
          all_metadata[f"{db_name}/{internal_key}"] = value
      return all_metadata

    if "/" not in key:
      raise ValueError(
          f"Metadata key {key} must be in format 'db_name/internal_key'"
      )
    db_name, internal_key = key.split("/", 1)
    if db_name not in self._dbs:
      raise ValueError(f"Database {db_name} not found in MultiDBWrapper")
    return self._dbs[db_name].get_metadata(internal_key)

  def remove_metadata(self, key: str | None) -> None:
    if key is None:
      for db in self._dbs_list:
        db.remove_metadata(None)
      return

    if "/" not in key:
      raise ValueError(
          f"Metadata key {key} must be in format 'db_name/internal_key'"
      )
    db_name, internal_key = key.split("/", 1)
    if db_name not in self._dbs:
      raise ValueError(f"Database {db_name} not found in MultiDBWrapper")
    self._dbs[db_name].remove_metadata(internal_key)

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
    deployment = self._dbs_list[db_index].get_deployment(internal_id)
    # The returned Deployment is the same as the internal Deployment, except
    # its ID is the external deployment_id argument of get_deployment().
    return deployment.replace(id=deployment_id)

  def remove_deployment(self, deployment_id: int) -> None:
    db_index, internal_id = self._split_id(deployment_id)
    self._dbs_list[db_index].remove_deployment(internal_id)

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
    internal_rec_id = self._dbs_list[db_index].insert_recording(
        filename=filename,
        datetime=datetime,
        deployment_id=internal_dep_id,
        **kwargs,
    )
    return self._join_id(db_index, internal_rec_id)

  def get_recording(self, recording_id: int) -> datatypes.Recording:
    db_index, internal_rec_id = self._split_id(recording_id)
    recording = self._dbs_list[db_index].get_recording(internal_rec_id)
    deployment_id = None
    if recording.deployment_id is not None:
      deployment_id = self._join_id(db_index, recording.deployment_id)
    return recording.replace(id=recording_id, deployment_id=deployment_id)

  def remove_recording(self, recording_id: int) -> None:
    db_index, internal_id = self._split_id(recording_id)
    self._dbs_list[db_index].remove_recording(internal_id)

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
    internal_win_id = self._dbs_list[db_index].insert_window(
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
    window = self._dbs_list[db_index].get_window(
        internal_win_id, include_embedding
    )
    return window.replace(
        id=window_id,
        recording_id=self._join_id(db_index, window.recording_id),
    )

  def get_window_annotations(
      self, window_id: int
  ) -> Sequence[datatypes.Annotation]:
    db_index, internal_win_id = self._split_id(window_id)
    annotations = self._dbs_list[db_index].get_window_annotations(
        internal_win_id
    )
    return [
        ann.replace(
            id=self._join_id(db_index, ann.id),
            recording_id=self._join_id(db_index, ann.recording_id),
        )
        for ann in annotations
    ]

  def get_embedding(self, window_id: int) -> np.ndarray:
    db_index, internal_win_id = self._split_id(window_id)
    return self._dbs_list[db_index].get_embedding(internal_win_id)

  def remove_window(self, window_id: int) -> None:
    db_index, internal_win_id = self._split_id(window_id)
    self._dbs_list[db_index].remove_window(internal_win_id)

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
    internal_ann_id = self._dbs_list[db_index].insert_annotation(
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
    annotation = self._dbs_list[db_index].get_annotation(internal_ann_id)
    return annotation.replace(
        id=annotation_id,
        recording_id=self._join_id(db_index, annotation.recording_id),
    )

  def remove_annotation(self, annotation_id: int) -> None:
    db_index, internal_ann_id = self._split_id(annotation_id)
    self._dbs_list[db_index].remove_annotation(internal_ann_id)

  def count_embeddings(self) -> int:
    return sum(db.count_embeddings() for db in self._dbs_list)

  def match_window_ids(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      annotations_filter: config_dict.ConfigDict | None = None,
      limit: int | None = None,
  ) -> Sequence[int]:
    windows = self.get_all_windows(
        include_embedding=False,
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        filter=windows_filter,
        annotations_filter=annotations_filter,
    )
    ids = [window.id for window in windows]
    if limit is not None:
      return ids[:limit]
    return ids

  def get_all_projects(self) -> Sequence[str]:
    projects = set()
    for db in self._dbs_list:
      projects.update(db.get_all_projects())
    return sorted(list(projects))

  def get_all_deployments(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Deployment]:
    pass_through_filter, post_filter = filter_utils.split_filter(filter, ["id"])
    candidates = {}
    for db_index, db in enumerate(self._dbs_list):
      deployments = db.get_all_deployments(filter=pass_through_filter)
      for deployment in deployments:
        deployment_id = self._join_id(db_index, deployment.id)
        candidates[deployment_id] = deployment.replace(id=deployment_id)

    if post_filter is None:
      return list(candidates.values())

    matched_ids = filter_utils.select_matching_keys(candidates, post_filter)
    return [
        candidates[deployment_id]
        for deployment_id in candidates
        if deployment_id in matched_ids
    ]

  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Recording]:
    pass_through_filter, post_filter = filter_utils.split_filter(
        filter, ["id", "deployment_id"]
    )
    candidates = {}
    for db_index, db in enumerate(self._dbs_list):
      recordings = db.get_all_recordings(filter=pass_through_filter)
      for recording in recordings:
        recording_id = self._join_id(db_index, recording.id)
        candidates[recording_id] = recording.replace(
            id=recording_id,
            deployment_id=(
                self._join_id(db_index, recording.deployment_id)
                if recording.deployment_id is not None
                else None
            ),
        )

    if post_filter is None:
      return list(candidates.values())

    matched_ids = filter_utils.select_matching_keys(candidates, post_filter)
    return [
        candidates[recording_id]
        for recording_id in candidates
        if recording_id in matched_ids
    ]

  def get_all_windows(
      self,
      include_embedding: bool = False,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
      annotations_filter: config_dict.ConfigDict | None = None,
  ) -> Sequence[datatypes.Window]:
    """Get all windows from the database."""
    deployments_pass_through_filter, deployments_post_filter = (
        filter_utils.split_filter(deployments_filter, ["id"])
    )
    recordings_pass_through_filter, recordings_post_filter = (
        filter_utils.split_filter(recordings_filter, ["id", "deployment_id"])
    )
    pass_through_filter, post_filter = filter_utils.split_filter(
        filter, ["id", "recording_id"]
    )
    annotations_pass_through_filter, annotations_post_filter = (
        filter_utils.split_filter(annotations_filter, ["id", "recording_id"])
    )

    candidates = {}
    for db_index, db in enumerate(self._dbs_list):
      windows = db.get_all_windows(
          include_embedding=include_embedding,
          deployments_filter=deployments_pass_through_filter,
          recordings_filter=recordings_pass_through_filter,
          filter=pass_through_filter,
          annotations_filter=annotations_pass_through_filter,
      )
      for window in windows:
        window_id = self._join_id(db_index, window.id)
        candidates[window_id] = window.replace(
            id=window_id,
            recording_id=self._join_id(db_index, window.recording_id),
        )

    matched_ids = set(candidates.keys())
    if post_filter:
      matched_ids &= filter_utils.select_matching_keys(candidates, post_filter)

    if deployments_post_filter:
      # Pass the full deployments_filter instead of deployments_post_filter to
      # make the get_all_deployments call more efficient since more deployments
      # could get filtered out earlier on in the function call
      matched_deployments = self.get_all_deployments(filter=deployments_filter)
      matched_deployment_ids = {
          deployment.id for deployment in matched_deployments
      }

      # Get recording_ids belonging to matched deployments
      matched_recordings = self.get_all_recordings(
          filter=config_dict.ConfigDict(
              {"isin": {"deployment_id": list(matched_deployment_ids)}}
          )
      )
      matched_recording_ids = {recording.id for recording in matched_recordings}
      matched_ids = {
          window_id
          for window_id in matched_ids
          if candidates[window_id].recording_id in matched_recording_ids
      }

    if recordings_post_filter:
      # Pass the full recordings_filter instead of recordings_post_filter to
      # make the get_all_recordings call more efficient since more recordings
      # could get filtered out earlier on in the function call
      matched_recordings = self.get_all_recordings(filter=recordings_filter)
      matched_recording_ids = {recording.id for recording in matched_recordings}
      matched_ids = {
          window_id
          for window_id in matched_ids
          if candidates[window_id].recording_id in matched_recording_ids
      }

    if annotations_post_filter:
      # Pass the full annotations_filter instead of annotations_post_filter to
      # make the get_all_annotations call more efficient since more annotations
      # could get filtered out earlier on in the function call
      matched_annotations = self.get_all_annotations(filter=annotations_filter)
      matched_recording_ids = {
          annotation.recording_id for annotation in matched_annotations
      }
      matched_ids = {
          window_id
          for window_id in matched_ids
          if candidates[window_id].recording_id in matched_recording_ids
      }

    return [
        candidates[window_id]
        for window_id in candidates
        if window_id in matched_ids
    ]

  def get_all_annotations(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Annotation]:
    pass_through_filter, post_filter = filter_utils.split_filter(
        filter, ["id", "recording_id"]
    )
    candidates = {}
    for db_index, db in enumerate(self._dbs_list):
      annotations = db.get_all_annotations(filter=pass_through_filter)
      for annotation in annotations:
        annotation_id = self._join_id(db_index, annotation.id)
        candidates[annotation_id] = annotation.replace(
            id=annotation_id,
            recording_id=self._join_id(db_index, annotation.recording_id),
        )

    if post_filter is None:
      return list(candidates.values())

    matched_ids = filter_utils.select_matching_keys(candidates, post_filter)
    return [
        candidates[annotation_id]
        for annotation_id in candidates
        if annotation_id in matched_ids
    ]

  def get_all_labels(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> Sequence[str]:
    labels = set()
    for db in self._dbs_list:
      labels.update(db.get_all_labels(label_type=label_type))
    return sorted(list(labels))

  def count_each_label(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> collections.Counter[str]:
    counts = collections.Counter()
    for db in self._dbs_list:
      counts.update(db.count_each_label(label_type=label_type))
    return counts

  def get_embedding_dim(self) -> int:
    return self._dbs_list[0].get_embedding_dim()

  def get_embedding_dtype(self) -> type[Any]:
    return self._dbs_list[0].get_embedding_dtype()
