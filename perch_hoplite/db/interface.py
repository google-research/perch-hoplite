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

"""Base interface for searchable embeddings databases."""

import abc
import collections
from collections.abc import Sequence
import dataclasses
import datetime as dt
import enum
from typing import Any

from ml_collections import config_dict
import numpy as np


@dataclasses.dataclass(init=False, repr=False, eq=False)
class DynamicInfo:
  """A base dataclass to handle both pre-defined and arbitrary attributes.

  DynamicInfo uses the underlying dataclass implementation to define the set of
  fields that can be set. It also keeps a dictionary of arbitrary attributes
  that are not part of the dataclass. This allows for handling both pre-defined
  and arbitrary attributes in a unified manner.

  Child classes can be defined as:

  ```python
  @dataclasses.dataclass(init=False, repr=False, eq=False)
  class CustomInfo(DynamicInfo):
    required_attr: int
    optional_attr: str | None = None
  ```

  Creating CustomInfo instances is as simple as:

  ```python
  info = CustomInfo(required_attr=1)
  info = CustomInfo(required_attr=1, optional_attr="foo")
  info = CustomInfo(required_attr=1, random_attr="bar")
  info = CustomInfo(required_attr=1, optional_attr="foo", random_attr="bar")
  ```

  Getting and setting attributes, no matter if they are pre-defined or
  arbitrary, can be done via the same interface:

  ```python
  info.required_attr
  info.optional_attr
  info.random_attr
  info.required_attr = 1
  info.optional_attr = "foo"
  info.random_attr = "bar"
  ```
  """

  def __init__(self, **kwargs) -> None:
    # Get the set of fields that are defined in the child dataclass.
    defined_fields = {f.name for f in dataclasses.fields(self)}

    # This will store the arbitrary attributes.
    self._dynamic_info: dict[str, Any] = {}

    # Keep track of fields we haven't seen yet.
    missing_fields = defined_fields.copy()

    # Iterate through all provided keyword arguments.
    for key, value in kwargs.items():
      if key in defined_fields:
        # If the kwarg is a defined field, set it as a normal attribute.
        setattr(self, key, value)
        missing_fields.remove(key)
      else:
        # If it's not a defined field, add it to our dynamic dict.
        self._dynamic_info[key] = value

    # After processing, check if any non-default fields are missing.
    missing_non_default_fields = [
        f.name
        for f in dataclasses.fields(self)
        if (
            f.name in missing_fields
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        )
    ]
    if missing_non_default_fields:
      missing_non_default_fields_str = (
          "'" + "', '".join(missing_non_default_fields) + "'"
      )
      raise TypeError(
          f"'{type(self).__name__}.__init__()' missing required keyword-only"
          f" arguments: {missing_non_default_fields_str}."
      )

    # Manually call `__post_init__` if the user has defined one in a subclass.
    if hasattr(self, "__post_init__"):
      self.__post_init__()

  def __getattr__(self, name: str) -> Any:
    if name in self._dynamic_info:
      return self._dynamic_info[name]
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'."
    )

  def __setattr__(self, name: str, value: Any) -> None:
    # Use the existence of `_dynamic_info` to know if we are initialized.
    defined_fields = {f.name for f in dataclasses.fields(self)}
    if "_dynamic_info" not in self.__dict__ or name in defined_fields:
      super().__setattr__(name, value)
    else:
      self._dynamic_info[name] = value

  def __repr__(self) -> str:
    defined_parts = []
    for f in dataclasses.fields(self):
      value = getattr(self, f.name, dataclasses.MISSING)
      if value is not dataclasses.MISSING:
        defined_parts.append(f"{f.name}={repr(value)}")

    dynamic_parts = [f"{k}={repr(v)}" for k, v in self._dynamic_info.items()]

    all_parts = defined_parts + dynamic_parts
    return f"{self.__class__.__name__}({', '.join(all_parts)})"

  def __eq__(self, other: object) -> bool:
    # Two objects can't be equal if they are not of the same type.
    if self.__class__ is not other.__class__:
      return False

    # Compare both defined fields and `_dynamic_info` dictionaries.
    self_defined_values = tuple(
        getattr(self, f.name) for f in dataclasses.fields(self)
    )
    other_defined_values = tuple(
        getattr(other, f.name) for f in dataclasses.fields(self)
    )
    return (
        self_defined_values == other_defined_values
        and hasattr(self, "_dynamic_info")
        and hasattr(other, "_dynamic_info")
        and self._dynamic_info == other._dynamic_info
    )

  def __getstate__(self) -> dict[str, Any]:
    return {
        "__dict__": self.__dict__.copy(),
        "_dynamic_info": self._dynamic_info.copy(),
    }

  def __setstate__(self, state: dict[str, Any]) -> None:
    self.__dict__.update(state["__dict__"])
    self._dynamic_info.update(state["_dynamic_info"])

  def to_kwargs(self, skip: Sequence[str] | None = None) -> dict[str, Any]:
    """Convert dataclass to a dictionary of keyword arguments.

    Args:
      skip: A sequence of attribute names to skip.

    Returns:
      A dictionary of keyword arguments that can be passed to the constructor to
      create an equivalent object.
    """

    kwargs = {
        f.name: getattr(self, f.name)
        for f in dataclasses.fields(self)
        if getattr(self, f.name) is not dataclasses.MISSING
    }
    if "_dynamic_info" in self.__dict__:
      kwargs.update(self._dynamic_info)
    if skip is not None:
      for key in skip:
        kwargs.pop(key, None)
    return kwargs


@dataclasses.dataclass(init=False, repr=False, eq=False)
class Deployment(DynamicInfo):
  """Deployment (i.e. site) info."""

  id: int
  name: str
  project: str
  latitude: float | None = None
  longitude: float | None = None


@dataclasses.dataclass(init=False, repr=False, eq=False)
class Recording(DynamicInfo):
  """Recording info."""

  id: int
  filename: str
  datetime: dt.datetime | None = None
  deployment_id: int | None = None


@dataclasses.dataclass(init=False, repr=False, eq=False)
class Window(DynamicInfo):
  """Window info."""

  id: int
  recording_id: int
  offsets: np.ndarray
  embedding: np.ndarray | None


class LabelType(enum.Enum):
  NEGATIVE = 0
  POSITIVE = 1


@dataclasses.dataclass(init=False, repr=False, eq=False)
class Annotation(DynamicInfo):
  """Annotation info."""

  id: int
  window_id: int
  label: str
  label_type: LabelType
  provenance: str


@dataclasses.dataclass
class HopliteConfig:
  """Config dataclass used to handle ConfigDict objects in Hoplite databases."""

  def to_config_dict(self) -> config_dict.ConfigDict:
    """Convert to a ConfigDict."""
    return config_dict.ConfigDict(dataclasses.asdict(self))

  @classmethod
  def from_config_dict(cls, config: config_dict.ConfigDict) -> "HopliteConfig":
    """Convert from a ConfigDict."""
    return cls(**config)


class HopliteDBInterface(abc.ABC):
  """Interface for searchable embeddings database with metadata.

  The database consists of a table of embedding windows, with a unique id for
  each window and some additional tables for linking the embedding window to
  recording-level and site-level metadata. Additionally, a key-value table of
  ConfigDict objects is used to store Hoplite-specific metadata associated with
  the database.

  The interface provides multiple methods for filtering metadata, each of them
  accepting one or more ConfigDict filters with the following structure:

    - eq => to test if given column is equal to given value
      - column1: value1
      - column2: None
    - neq => to test if given column is not equal to given value
      - column1: value1
      - column2: None
    - lt => to test if given column is less than given value
      - column: value
    - lte => to test if given column is less than or equal to given value
      - column: value
    - gt => to test if given column is greater than given value
      - column: value
    - gte => to test if given column is greater than or equal to given value
      - column: value
    - isin => to test if given column is in given list of values
      - column: [value1, value2, value3, ...]
    - notin => to test if given column is not in given list of values
      - column: [value1, value2, value3, ...]
    - range => to test if given column is between two values
      - column: [value1, value2]

  The recommended way to build such ConfigDict filters is to use something like
  this (feel free to omit operations that are not needed):

  ```python
  from ml_collections import config_dict
  filter_dict = config_dict.create(
      eq=dict(column1=value1, column2=None),
      neq=dict(column1=value1, column2=None),
      lt=dict(column=value),
      lte=dict(column=value),
      gt=dict(column=value),
      gte=dict(column=value),
      isin=dict(column=[value1, value2, value3]),
      notin=dict(column=[value1, value2, value3]),
      range=dict(column=[value1, value2]),
  )
  ```
  """

  @classmethod
  @abc.abstractmethod
  def create(cls, **kwargs) -> "HopliteDBInterface":
    """Connect to and, if needed, initialize the database.

    Args:
      **kwargs: Keyword arguments to pass to the implementation-specific
        `create` function.

    Returns:
      A new instance of the database.
    """

  @abc.abstractmethod
  def add_extra_table_column(
      self,
      table_name: str,
      column_name: str,
      column_type: type[Any],
  ) -> None:
    """Add an extra column to a table in the database.

    Args:
      table_name: The name of the table to add the column to.
      column_name: The name of the column to add.
      column_type: The type of the column to add.
    """

  @abc.abstractmethod
  def get_extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    """Get all extra columns in the database."""

  @abc.abstractmethod
  def commit(self) -> None:
    """Commit any pending transactions to the database."""

  @abc.abstractmethod
  def thread_split(self) -> "HopliteDBInterface":
    """Get a new instance of the database with the same contents.

    For example, SQLite databases need a distinct object in each thread.

    Returns:
      A new instance of the database.
    """

  @abc.abstractmethod
  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    """Insert a key-value pair into the metadata table.

    Args:
      key: String for metadata key.
      value: ConfigDict object to store.
    """

  @abc.abstractmethod
  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    """Get a key-value pair from the metadata table.

    Args:
      key: String for metadata key to retrieve. If None, returns all metadata.

    Returns:
      A ConfigDict containing the metadata.
    """

  @abc.abstractmethod
  def remove_metadata(self, key: str | None) -> None:
    """Remove a key-value pair from the metadata table.

    Args:
      key: String for metadata key to remove. If None, removes all metadata.
    """

  @abc.abstractmethod
  def insert_deployment(
      self,
      name: str,
      project: str,
      latitude: float | None = None,
      longitude: float | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a deployment into the database.

    Args:
      name: The name of the deployment.
      project: The project of the deployment.
      latitude: The (optional) latitude of the deployment.
      longitude: The (optional) longitude of the deployment.
      **kwargs: Additional keyword arguments to pass to the deployment.

    Returns:
      The ID of the inserted deployment.
    """

  @abc.abstractmethod
  def get_deployment(self, deployment_id: int) -> Deployment:
    """Get a deployment from the database.

    Args:
      deployment_id: The ID of the deployment to retrieve.

    Returns:
      A Deployment object containing the requested information.
    """

  @abc.abstractmethod
  def remove_deployment(self, deployment_id: int) -> None:
    """Remove a deployment from the database.

    Args:
      deployment_id: The ID of the deployment to remove.
    """

  @abc.abstractmethod
  def insert_recording(
      self,
      filename: str,
      datetime: dt.datetime | None = None,
      deployment_id: int | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a recording into the database.

    Args:
      filename: The filename of the recording.
      datetime: The (optional) datetime of the recording.
      deployment_id: The (optional) ID of the deployment to which the recording
        belongs.
      **kwargs: Additional keyword arguments to pass to the recording.

    Returns:
      The ID of the inserted recording.
    """

  @abc.abstractmethod
  def get_recording(self, recording_id: int) -> Recording:
    """Get a recording from the database.

    Args:
      recording_id: The ID of the recording to retrieve.

    Returns:
      A Recording object containing the requested information.
    """

  @abc.abstractmethod
  def remove_recording(self, recording_id: int) -> None:
    """Remove a recording from the database.

    Args:
      recording_id: The ID of the recording to remove.
    """

  @abc.abstractmethod
  def insert_window(
      self,
      recording_id: int,
      offsets: np.ndarray,
      embedding: np.ndarray | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a window into the database.

    Args:
      recording_id: The ID of the recording to which the window belongs.
      offsets: The offsets of the window.
      embedding: The embedding vector. If None, no embedding vector is inserted
        into the database for this particular window.
      **kwargs: Additional keyword arguments to pass to the window.

    Returns:
      The ID of the inserted window.
    """

  @abc.abstractmethod
  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> Window:
    """Get a window from the database.

    Args:
      window_id: The ID of the window to retrieve.
      include_embedding: Whether to include the embedding vector in the returned
        Window object.

    Returns:
      A Window object containing the requested information.
    """

  @abc.abstractmethod
  def get_embedding(self, window_id: int) -> np.ndarray:
    """Get an embedding vector from the database.

    Args:
      window_id: The window ID of the embedding to retrieve.

    Returns:
      An embedding vector for the given window ID.
    """

  def get_embeddings_batch(
      self,
      window_ids: Sequence[int] | np.ndarray,
  ) -> np.ndarray:
    """Get a batch of embedding vectors from the database.

    Args:
      window_ids: The window IDs of the embeddings to retrieve.

    Returns:
      A batch of embedding vectors for the given window IDs.
    """

    embeddings = [self.get_embedding(id) for id in window_ids]
    return np.stack(embeddings)

  @abc.abstractmethod
  def remove_window(self, window_id: int) -> None:
    """Remove a window from the database.

    Args:
      window_id: The ID of the window to remove.
    """

  @abc.abstractmethod
  def insert_annotation(
      self,
      window_id: int,
      label: str,
      label_type: LabelType,
      provenance: str,
      skip_duplicates: bool = False,
      **kwargs: Any,
  ) -> int:
    """Insert an annotation into the database.

    Args:
      window_id: The ID of the window to which the annotation points.
      label: The annotation label.
      label_type: The type of label (e.g. positive or negative).
      provenance: The provenance of the annotation.
      skip_duplicates: If True and another annotation with the same (window_id,
        label, label_type) already exists in the database, return the id (or one
        of the ids) of that matching annotation. Otherwise, the annotation is
        inserted regardless of duplicates.
      **kwargs: Additional keyword arguments to pass to the annotation.

    Returns:
      The ID of the inserted annotation, or of the matching annotation if it
      already exists and `skip_duplicates` is True.
    """

  @abc.abstractmethod
  def get_annotation(self, annotation_id: int) -> Annotation:
    """Get an annotation from the database.

    Args:
      annotation_id: The ID of the annotation to retrieve.

    Returns:
      An Annotation object containing the requested information.
    """

  @abc.abstractmethod
  def remove_annotation(self, annotation_id: int) -> None:
    """Remove an annotation from the database.

    Args:
      annotation_id: The ID of the annotation to remove.
    """

  @abc.abstractmethod
  def count_embeddings(self) -> int:
    """Get the number of embeddings in the database."""

  @abc.abstractmethod
  def match_window_ids(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      annotations_filter: config_dict.ConfigDict | None = None,
      limit: int | None = None,
  ) -> Sequence[int]:
    """Get matching window IDs from the database based on given filters.

    Args:
      deployments_filter: If provided, only retrieve window IDs that have
        deployments matching constraints specified by this filter.
      recordings_filter: If provided, only retrieve window IDs that have
        recordings matching constraints specified by this filter.
      windows_filter: If provided, only retrieve window IDs that have windows
        matching constraints specified by this filter.
      annotations_filter: If provided, only retrieve window IDs that have
        annotations matching constraints specified by this filter.
      limit: If provided, limit the number of window IDs returned.

    Returns:
      A sequence of window IDs from the database.
    """

  @abc.abstractmethod
  def get_all_projects(self) -> Sequence[str]:
    """Get all distinct projects from the database.

    Returns:
      A sequence of all projects in the database.
    """

  @abc.abstractmethod
  def get_all_deployments(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[Deployment]:
    """Get all deployments from the database.

    Args:
      filter: If provided, only retrieve deployments matching constraints
        specified by this filter.

    Returns:
      A sequence of all matching deployments in the database.
    """

  @abc.abstractmethod
  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[Recording]:
    """Get all recordings from the database.

    Args:
      filter: If provided, only retrieve recordings matching constraints
        specified by this filter.

    Returns:
      A sequence of all matching recordings in the database.
    """

  @abc.abstractmethod
  def get_all_windows(
      self,
      include_embedding: bool = False,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[Window]:
    """Get all windows from the database.

    Args:
      include_embedding: Whether to include the embedding vector in the returned
        Window objects.
      filter: If provided, only retrieve windows matching constraints specified
        by this filter.

    Returns:
      A sequence of all matching windows in the database.
    """

  @abc.abstractmethod
  def get_all_annotations(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[Annotation]:
    """Get all annotations from the database.

    Args:
      filter: If provided, only retrieve annotations matching constraints
        specified by this filter.

    Returns:
      A sequence of all matching annotations in the database.
    """

  @abc.abstractmethod
  def get_all_labels(
      self,
      label_type: LabelType | None = None,
  ) -> Sequence[str]:
    """Get all distinct labels from the database.

    Args:
      label_type: If provided, filters to the target label type.

    Returns:
      A sequence of all labels in the database.
    """

  @abc.abstractmethod
  def count_each_label(
      self,
      label_type: LabelType | None = None,
  ) -> collections.Counter[str]:
    """Count each label in the database, ignoring provenance.

    Args:
      label_type: If provided, filters to the target label type.

    Returns:
      A counter dict for labels.
    """

  @abc.abstractmethod
  def get_embedding_dim(self) -> int:
    """Get the embedding dimension."""

  @abc.abstractmethod
  def get_embedding_dtype(self) -> type[Any]:
    """Get the embedding data type."""
