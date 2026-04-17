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

"""Utility functions regarding filtering for Hoplite databases."""

from collections.abc import Sequence
from typing import Any

from absl import logging
from ml_collections import config_dict
import numpy as np


def select_matching_keys(
    kv: dict[int, Any],
    filter_dict: config_dict.ConfigDict | None = None,
) -> set[int]:
  """Select the keys from a dictionary whose values match the given constraints.

  Args:
    kv: The dictionary to verify against the constraints.
    filter_dict: An optional ConfigDict of constraints to verify.

  Returns:
    The keys of the matching items.
  """

  if not filter_dict:
    return set(kv.keys())

  supported_ops = {
      'eq',
      'neq',
      'lt',
      'lte',
      'gt',
      'gte',
      'isin',
      'notin',
      'range',
      'approx',
  }
  for op_name, op_filters in filter_dict.items():
    if op_name not in supported_ops:
      raise ValueError(
          f'Unsupported operation: `{op_name}`. Supported filtering operations'
          f' are: {supported_ops}.'
      )
    if not isinstance(op_filters, config_dict.ConfigDict):
      raise ValueError(f'`{op_name}` value must be a ConfigDict.')

  def _is_match(obj: Any) -> bool:
    for op_name, op_filters in filter_dict.items():
      for key, value in op_filters.items():
        attr = getattr(obj, key, None)

        if op_name == 'eq':
          if key == 'offsets':
            logging.warning(
                "Do not apply `eq` to the `offsets` unless you know what you're"
                ' doing. Apply `approx` instead to avoid floating point errors.'
            )
          if attr is None:
            if value is not None:
              return False
          else:
            if isinstance(attr, np.ndarray):
              if (attr != value).any():
                return False
            else:
              if attr != value:
                return False
        elif op_name == 'neq':
          if attr is None:
            if value is None:
              return False
          else:
            if isinstance(attr, np.ndarray):
              if (attr != value).all():
                return False
            else:
              if attr == value:
                return False
        elif op_name == 'lt':
          if attr is None or value is None:
            return False
          else:
            if attr >= value:
              return False
        elif op_name == 'lte':
          if attr is None or value is None:
            return False
          else:
            if attr > value:
              return False
        elif op_name == 'gt':
          if attr is None or value is None:
            return False
          else:
            if attr <= value:
              return False
        elif op_name == 'gte':
          if attr is None or value is None:
            return False
          else:
            if attr < value:
              return False
        elif op_name == 'isin':
          if not isinstance(value, list):
            raise ValueError(f'`{op_name}` value must be a list.')
          if attr not in value:
            return False
        elif op_name == 'notin':
          if not isinstance(value, list):
            raise ValueError(f'`{op_name}` value must be a list.')
          if attr in value:
            return False
        elif op_name == 'range':
          if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f'`{op_name}` value must be a list of 2 elements.')
          if attr is None or attr < value[0] or attr > value[1]:
            return False
        elif op_name == 'approx':
          if attr is None or value is None:
            return False
          if key == 'offsets':
            if not np.allclose(attr, value, rtol=0.0, atol=1e-6):
              return False
          elif abs(attr - value) >= 1e-6:
            return False

    return True

  return {key for key, value in kv.items() if _is_match(value)}


def split_filter(
    filter_dict: config_dict.ConfigDict | None,
    column_names_in_second_filter: Sequence[str],
) -> tuple[config_dict.ConfigDict | None, config_dict.ConfigDict | None]:
  """Split a filter into two based on column names.

  Args:
    filter_dict: The original filter of constraints.
    column_names_in_second_filter: The column names that should be split off
      into the second returned filter.

  Returns:
    A tuple (first_filter, second_filter).
  """
  if not filter_dict:
    return None, None
  first_filter = config_dict.ConfigDict()
  second_filter = config_dict.ConfigDict()
  for op_name, op_filters in filter_dict.items():
    first_op_filters = config_dict.ConfigDict()
    second_op_filters = config_dict.ConfigDict()
    for key, value in op_filters.items():
      if key in column_names_in_second_filter:
        second_op_filters[key] = value
      else:
        first_op_filters[key] = value
    if first_op_filters:
      first_filter[op_name] = first_op_filters
    if second_op_filters:
      second_filter[op_name] = second_op_filters
  return (
      first_filter if first_filter else None,
      second_filter if second_filter else None,
  )
