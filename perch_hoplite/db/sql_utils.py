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

"""Shared SQL formatting and validation utilities for Hoplite DB backends."""

from collections.abc import Callable, Sequence
import datetime as dt
import itertools
import re
from typing import Any

from absl import logging
from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes

_SQL_IDENTIFIER_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

SUPPORTED_FILTER_OPS = frozenset({
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
})

OffsetsApproxFn = Callable[[str, Any], tuple[str, Sequence[Any]]]


def is_valid_sql_identifier(name: str) -> bool:
  """Check if a string is a valid and safe unquoted SQL identifier."""
  if not name or not isinstance(name, str):
    return False
  return _SQL_IDENTIFIER_RE.match(name) is not None


def quote_sql_identifier(name: str) -> str:
  """Double-quote a SQL identifier, escaping any embedded double quotes.

  Useful for caller-supplied identifiers such as class labels that may contain
  spaces, hyphens, or punctuation (e.g. "American Redstart").

  Args:
    name: The identifier string to quote.

  Returns:
    A safely double-quoted SQL identifier.
  """
  return '"' + name.replace('"', '""') + '"'


def normalize_sql_value(value: Any) -> Any:
  """Normalize a Python value to a type supported by SQL backends."""
  if isinstance(value, (list, tuple, np.ndarray)):
    return [normalize_sql_value(v) for v in value]
  if isinstance(value, datatypes.LabelType):
    return value.value
  if isinstance(value, dt.datetime):
    return value.isoformat()
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.floating):
    return float(value)
  return value


def format_sql_insert_values(
    **kwargs: Any,
) -> tuple[str, str, list[Any]]:
  """Build columns string, placeholders string, and values list for SQL INSERT.

  Args:
    **kwargs: Key-value pairs to pass to the SQL statement.

  Returns:
    A tuple of: a formatted columns string, a formatted placeholders string, and
    a list of corresponding values. Safe to be used in SQL INSERT statements.

  Raises:
    ValueError: If any key is not a valid SQL identifier.
  """
  for key in kwargs:
    if not is_valid_sql_identifier(key):
      raise ValueError(f'`{key}` is not a valid SQL identifier.')

  columns = list(kwargs.keys())
  placeholders = ['?'] * len(columns)
  values = normalize_sql_value(list(kwargs.values()))

  return f"({', '.join(columns)})", f"({', '.join(placeholders)})", values


def format_sql_update_on_conflict(*args: str) -> str:
  """Build the update part of ON CONFLICT clauses for INSERT statements.

  Args:
    *args: Column names to update on conflict.

  Returns:
    Either `'DO NOTHING'` (if no columns are provided) or a `'DO UPDATE SET'`
    clause updating each column from `excluded`.

  Raises:
    ValueError: If any column name is not a valid SQL identifier.
  """
  for key in args:
    if not is_valid_sql_identifier(key):
      raise ValueError(f'`{key}` is not a valid SQL identifier.')

  if not args:
    return 'DO NOTHING'
  update_clauses_str = ', '.join(f'{key} = excluded.{key}' for key in args)
  return f'DO UPDATE SET {update_clauses_str}'


def _default_sqlite_offsets_approx(
    column: str, value: Any
) -> tuple[str, Sequence[Any]]:
  return f'APPROX_FLOAT_LIST({column}, ?) = TRUE', (value,)


def format_sql_where_conditions(
    filter_dict: config_dict.ConfigDict | None = None,
    table_prefix: str | None = None,
    *,
    offsets_approx_fn: OffsetsApproxFn | None = None,
) -> tuple[str, list[Any]]:
  """Build conditions string and values list for SQL WHERE from given filters.

  Args:
    filter_dict: A ConfigDict of constraints to build SQL conditions from. If
      None or empty, returns an empty condition string and empty values list.
    table_prefix: An optional table prefix to prepend to each column name.
    offsets_approx_fn: Optional callable `(column, value) -> (sql_expr, params)`
      for dialect-specific approximate matching on `offsets`. Defaults to
      SQLite's `APPROX_FLOAT_LIST({column}, ?) = TRUE`.

  Returns:
    A tuple of: a formatted string of AND-joined conditions, and a list of
    corresponding values. Safe to be used in SQL WHERE statements.

  Raises:
    ValueError: If `table_prefix`, an operation name, a column name, or a filter
      value has an invalid format.
  """
  if table_prefix and not is_valid_sql_identifier(table_prefix):
    raise ValueError(
        f'Table prefix `{table_prefix}` is not a valid SQL identifier.'
    )

  if not filter_dict:
    return '', []

  if offsets_approx_fn is None:
    offsets_approx_fn = _default_sqlite_offsets_approx

  conditions = []
  values = []

  for op_name, op_filters in filter_dict.items():
    if op_name not in SUPPORTED_FILTER_OPS:
      raise ValueError(
          f'Unsupported operation: `{op_name}`. Supported filtering operations'
          f' are: {sorted(SUPPORTED_FILTER_OPS)}.'
      )
    if not isinstance(op_filters, config_dict.ConfigDict):
      raise ValueError(f'`{op_name}` value must be a ConfigDict.')

    for key, value in op_filters.items():
      column = f'{table_prefix}.{key}' if table_prefix else key

      if not is_valid_sql_identifier(key):
        raise ValueError(
            f'Table column `{column}` is not a valid SQL identifier. Fix the'
            f' filter rule under the `{op_name}` operation.'
        )

      value = normalize_sql_value(value)

      if op_name == 'eq':
        if key == 'offsets':
          logging.warning(
              "Do not apply `eq` to the `offsets` unless you know what you're "
              'doing. Apply `approx` instead to avoid floating point errors.'
          )
        if value is None:
          conditions.append(f'{column} IS NULL')
        else:
          conditions.append(f'{column} = ?')
          values.append(value)
      elif op_name == 'neq':
        if value is None:
          conditions.append(f'{column} IS NOT NULL')
        else:
          conditions.append(f'{column} != ?')
          values.append(value)
      elif op_name == 'lt':
        conditions.append(f'{column} < ?')
        values.append(value)
      elif op_name == 'lte':
        conditions.append(f'{column} <= ?')
        values.append(value)
      elif op_name == 'gt':
        conditions.append(f'{column} > ?')
        values.append(value)
      elif op_name == 'gte':
        conditions.append(f'{column} >= ?')
        values.append(value)
      elif op_name == 'isin':
        if not isinstance(value, list):
          raise ValueError(f'`{op_name}` value must be a list.')
        if not value:
          conditions.append('1 = 0')
        else:
          placeholders_str = ', '.join(['?'] * len(value))
          conditions.append(f'{column} IN ({placeholders_str})')
          values.extend(value)
      elif op_name == 'notin':
        if not isinstance(value, list):
          raise ValueError(f'`{op_name}` value must be a list.')
        if not value:
          conditions.append('1 = 1')
        else:
          placeholders_str = ', '.join(['?'] * len(value))
          conditions.append(f'{column} NOT IN ({placeholders_str})')
          values.extend(value)
      elif op_name == 'range':
        if not isinstance(value, list) or len(value) != 2:
          raise ValueError(f'`{op_name}` value must be a list of 2 elements.')
        conditions.append(f'{column} BETWEEN ? AND ?')
        values.extend(value)
      elif op_name == 'approx':
        if key == 'offsets':
          expr, params = offsets_approx_fn(column, value)
          conditions.append(expr)
          values.extend(params)
        else:
          conditions.append(f'ABS({column} - ?) < 1e-6')
          values.append(value)

  return ' AND '.join(conditions), values


def get_window_query_components(
    deployments_filter: config_dict.ConfigDict | None = None,
    recordings_filter: config_dict.ConfigDict | None = None,
    windows_filter: config_dict.ConfigDict | None = None,
    annotations_filter: config_dict.ConfigDict | None = None,
    *,
    offset_start_template: str = 'GET_OFFSET_START({column})',
    offset_end_template: str = 'GET_OFFSET_END({column})',
    offsets_approx_fn: OffsetsApproxFn | None = None,
) -> tuple[str, str, list[Any]]:
  """Construct FROM, WHERE, and VALUES for window SQL queries.

  Args:
    deployments_filter: Optional filter on the `deployments` table.
    recordings_filter: Optional filter on the `recordings` table.
    windows_filter: Optional filter on the `windows` table.
    annotations_filter: Optional filter on the `annotations` table.
    offset_start_template: Format template receiving `column` to extract the
      start offset in SQL.
    offset_end_template: Format template receiving `column` to extract the end
      offset in SQL.
    offsets_approx_fn: Optional dialect-specific `approx` handler for `offsets`.

  Returns:
    A tuple `(from_clause, where_clause, values)`.
  """
  query_tables = {'windows'}
  if annotations_filter:
    query_tables |= {'annotations'}
  if recordings_filter:
    query_tables |= {'recordings'}
  if deployments_filter:
    query_tables |= {'recordings', 'deployments'}

  from_clause = 'FROM windows'
  if 'annotations' in query_tables:
    ann_start = offset_start_template.format(column='annotations.offsets')
    ann_end = offset_end_template.format(column='annotations.offsets')
    win_start = offset_start_template.format(column='windows.offsets')
    win_end = offset_end_template.format(column='windows.offsets')
    from_clause += (
        ' JOIN annotations ON windows.recording_id = annotations.recording_id'
        f' AND {ann_start} < {win_end}'
        f' AND {ann_end} > {win_start}'
    )
  if 'recordings' in query_tables:
    from_clause += ' JOIN recordings ON windows.recording_id = recordings.id'
  if 'deployments' in query_tables:
    from_clause += (
        ' JOIN deployments ON recordings.deployment_id = deployments.id'
    )

  conditions, values = tuple(
      zip(*[
          format_sql_where_conditions(
              deployments_filter,
              table_prefix='deployments',
              offsets_approx_fn=offsets_approx_fn,
          ),
          format_sql_where_conditions(
              recordings_filter,
              table_prefix='recordings',
              offsets_approx_fn=offsets_approx_fn,
          ),
          format_sql_where_conditions(
              windows_filter,
              table_prefix='windows',
              offsets_approx_fn=offsets_approx_fn,
          ),
          format_sql_where_conditions(
              annotations_filter,
              table_prefix='annotations',
              offsets_approx_fn=offsets_approx_fn,
          ),
      ])
  )
  conditions_str = ' AND '.join(c for c in conditions if c)
  flat_values = list(itertools.chain.from_iterable(values))
  where_clause = f'WHERE {conditions_str}' if conditions_str else ''
  return from_clause, where_clause, flat_values
