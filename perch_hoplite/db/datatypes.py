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

"""Data classes for Hoplite databases."""

import dataclasses
import datetime as dt
import enum
from typing import Any, Sequence

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
  offsets: list[float]
  embedding: np.ndarray | None


class LabelType(int, enum.Enum):
  NEGATIVE = 0
  POSITIVE = 1
  UNCERTAIN = 2


@dataclasses.dataclass(init=False, repr=False, eq=False)
class Annotation(DynamicInfo):
  """Annotation info."""

  id: int
  recording_id: int
  offsets: list[float]
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
