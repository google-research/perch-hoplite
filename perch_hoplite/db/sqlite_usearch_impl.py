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

"""SQLite database implementation using USearch for vector storage & search."""

import collections
from collections.abc import Sequence
import dataclasses
import datetime as dt
import functools
import itertools
import json
import re
import sqlite3
import threading
from typing import Any, Literal

from absl import logging
from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import brutalism
from perch_hoplite.db import datatypes
from perch_hoplite.db import interface
from perch_hoplite.db import score_functions
from perch_hoplite.db import search_results
from usearch import index as uindex

HOPLITE_FILENAME = 'hoplite.sqlite'
UINDEX_FILENAME = 'usearch.index'
USEARCH_CONFIG_KEY = 'usearch_config'
USEARCH_DTYPES = {
    'float16': uindex.ScalarKind.F16,
}
SQL_TYPE_TO_PYTHON_TYPE = {
    'INTEGER': int,
    'REAL': float,
    'TEXT': str,
    'BLOB': bytes,
    'TIMESTAMP': dt.datetime,
    'FLOAT_LIST': list,
}
PYTHON_TYPE_TO_SQL_TYPE = {
    int: 'INTEGER',
    float: 'REAL',
    str: 'TEXT',
    bytes: 'BLOB',
    dt.datetime: 'TEXT',
    list: 'FLOAT_LIST',
}


def adapt_float_list(data: list[float]) -> bytes:
  return np.array(
      data,
      dtype=np.dtype('<f8'),  # little-endian np.float64
  ).tobytes()


def convert_float_list(blob: bytes) -> list[float]:
  return np.frombuffer(
      blob,
      dtype=np.dtype('<f8'),  # little-endian np.float64
  ).tolist()


def approx_float_list(blob: bytes, target: bytes) -> bool:
  return np.allclose(
      convert_float_list(blob),
      convert_float_list(target),
      rtol=0.0,
      atol=1e-6,
  )


def get_offset_start(blob: bytes) -> float:
  """Extract start offset from blob."""
  return np.frombuffer(
      blob,
      dtype=np.dtype('<f8'),  # little-endian np.float64
  )[0]


def get_offset_end(blob: bytes) -> float:
  """Extract end offset from blob."""
  return np.frombuffer(
      blob,
      dtype=np.dtype('<f8'),  # little-endian np.float64
  )[1]


def convert_timestamp(val: bytes) -> dt.datetime | None:
  if not val:
    return None
  return dt.datetime.fromisoformat(val.decode('utf-8'))


sqlite3.register_adapter(list, adapt_float_list)
sqlite3.register_converter('FLOAT_LIST', convert_float_list)
sqlite3.register_converter('TIMESTAMP', convert_timestamp)


def get_default_usearch_config(
    embedding_dim: int,
) -> config_dict.ConfigDict:
  """Get a default USearch config for the given embedding dimension."""
  usearch_cfg = config_dict.ConfigDict()
  usearch_cfg.embedding_dim = embedding_dim
  usearch_cfg.dtype = 'float16'
  usearch_cfg.metric_name = 'IP'
  usearch_cfg.expansion_add = 256
  usearch_cfg.expansion_search = 128
  return usearch_cfg


def is_valid_sql_identifier(name: str) -> bool:
  """Check if a string is a valid and safe SQL identifier."""

  if not name or not isinstance(name, str):
    return False

  # Regex to verify that the name starts with a letter or underscore, then
  # follows with letters, numbers or underscores.
  return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None


def normalize_sql_value(value: Any) -> Any:
  """Normalize a python value to one of the types supported by SQL."""

  if (
      isinstance(value, list)
      or isinstance(value, tuple)
      or isinstance(value, np.ndarray)
  ):
    return [normalize_sql_value(v) for v in value]
  if isinstance(value, datatypes.LabelType):
    return value.value
  elif isinstance(value, dt.datetime):
    return value.isoformat()
  elif isinstance(value, np.integer):
    return int(value)
  elif isinstance(value, np.floating):
    return float(value)
  return value


def format_sql_insert_values(
    **kwargs: Any,
) -> tuple[str, str, list[Any]]:
  """Build columns string, placeholders string and values list for SQL INSERT.

  Args:
    **kwargs: Key-value pairs to pass to the SQL statement.

  Returns:
    A tuple of: a formatted columns string, a formatted placeholders string, and
    a list of corresponding values. Safe to be used in SQL INSERT statements.
  """

  for key in kwargs:
    if not is_valid_sql_identifier(key):
      raise ValueError(f'`{key}` is not a valid SQL identifier.')

  columns = list(kwargs.keys())
  placeholders = ['?'] * len(columns)
  values = normalize_sql_value(list(kwargs.values()))

  return f"({', '.join(columns)})", f"({', '.join(placeholders)})", values


def format_sql_update_on_conflict(*args: str) -> str:
  """Build the update part of ON CONFLICT clauses for INSERT statements."""

  for key in args:
    if not is_valid_sql_identifier(key):
      raise ValueError(f'`{key}` is not a valid SQL identifier.')

  if not args:
    return 'DO NOTHING'
  else:
    update_clauses_str = ', '.join([f'{key} = excluded.{key}' for key in args])
    return f'DO UPDATE SET {update_clauses_str}'


def format_sql_where_conditions(
    filter_dict: config_dict.ConfigDict | None = None,
    table_prefix: str | None = None,
) -> tuple[str, list[Any]]:
  r"""Build conditions string and values list for SQL WHERE from given filters.

  Args:
    filter_dict: A ConfigDict of constraints to build SQL conditions from.
    table_prefix: An optional table prefix to prepend to each column name.

  Returns:
    A tuple of: a formatted string of AND-joined conditions, and a list of
    corresponding values. Safe to be used in SQL WHERE statements.
  """

  if table_prefix and not is_valid_sql_identifier(table_prefix):
    raise ValueError(
        f'Table prefix `{table_prefix}` is not a valid SQL identifier.'
    )

  if not filter_dict:
    return '', []

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

  conditions = []
  values = []

  # Build the SQL conditions for each operation.
  for op_name, op_filters in filter_dict.items():

    if op_name not in supported_ops:
      raise ValueError(
          f'Unsupported operation: `{op_name}`. Supported filtering operations'
          f' are: {supported_ops}.'
      )
    if not isinstance(op_filters, config_dict.ConfigDict):
      raise ValueError(f'`{op_name}` value must be a ConfigDict.')

    for key, value in op_filters.items():
      if table_prefix:
        column = f'{table_prefix}.{key}'
      else:
        column = key

      # Check that the key is a valid SQL identifier.
      if not is_valid_sql_identifier(key):
        raise ValueError(
            f'Table column `{column}` is not a valid SQL identifier. Fix the'
            f' filter rule under the `{op_name}` operation.'
        )

      # Normalize the value.
      value = normalize_sql_value(value)

      # Build the current SQL condition.
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
        placeholders = ['?'] * len(value)
        placeholders_str = ', '.join(placeholders)
        conditions.append(f'{column} IN ({placeholders_str})')
        values.extend(value)
      elif op_name == 'notin':
        if not isinstance(value, list):
          raise ValueError(f'`{op_name}` value must be a list.')
        placeholders = ['?'] * len(value)
        placeholders_str = ', '.join(placeholders)
        conditions.append(f'{column} NOT IN ({placeholders_str})')
        values.extend(value)
      elif op_name == 'range':
        if not isinstance(value, list) or len(value) != 2:
          raise ValueError(f'`{op_name}` value must be a list of 2 elements.')
        conditions.append(f'{column} BETWEEN ? AND ?')
        values.extend(value)
      elif op_name == 'approx':
        if key == 'offsets':
          conditions.append(f'APPROX_FLOAT_LIST({column}, ?) = TRUE')
        else:
          conditions.append(f'ABS({column} - ?) < 1e-6')
        values.append(value)

  return ' AND '.join(conditions), values


def _get_window_query_components(
    deployments_filter: config_dict.ConfigDict | None = None,
    recordings_filter: config_dict.ConfigDict | None = None,
    windows_filter: config_dict.ConfigDict | None = None,
    annotations_filter: config_dict.ConfigDict | None = None,
) -> tuple[str, str, list[Any]]:
  """Construct FROM, WHERE, and VALUES for window SQL queries."""
  # Pick which tables need to be queried.
  query_tables = {'windows'}
  if annotations_filter:
    query_tables |= {'annotations'}
  if recordings_filter:
    query_tables |= {'recordings'}
  if deployments_filter:
    query_tables |= {'recordings', 'deployments'}

  # Build the `FROM ... [JOIN ...]` part of the SQL query.
  from_clause = 'FROM windows'
  if 'annotations' in query_tables:
    from_clause += (
        ' JOIN annotations ON windows.recording_id = annotations.recording_id'
        ' AND GET_OFFSET_START(annotations.offsets) <'
        ' GET_OFFSET_END(windows.offsets)'
        ' AND GET_OFFSET_END(annotations.offsets) >'
        ' GET_OFFSET_START(windows.offsets)'
    )
  if 'recordings' in query_tables:
    from_clause += ' JOIN recordings ON windows.recording_id = recordings.id'
  if 'deployments' in query_tables:
    from_clause += (
        ' JOIN deployments ON recordings.deployment_id = deployments.id'
    )

  # Build the `WHERE ...` part of the SQL query.
  conditions, values = tuple(
      zip(*[
          format_sql_where_conditions(
              deployments_filter, table_prefix='deployments'
          ),
          format_sql_where_conditions(
              recordings_filter, table_prefix='recordings'
          ),
          format_sql_where_conditions(windows_filter, table_prefix='windows'),
          format_sql_where_conditions(
              annotations_filter, table_prefix='annotations'
          ),
      ])
  )
  conditions_str = ' AND '.join(c for c in conditions if c)
  values = list(itertools.chain.from_iterable(values))
  where_clause = f'WHERE {conditions_str}' if conditions_str else ''
  return from_clause, where_clause, values


@dataclasses.dataclass
class SQLiteUSearchDB(interface.HopliteDBInterface):
  """SQLite hoplite database, using USearch for vector storage.

  USearch provides both indexing for approximate nearest neighbor search and
  fast disk-based random access to vectors for the complete database. USearch
  will default to working with disk-based vectors, unless we insert or remove
  embeddings, in which case we load the index into memory and use it from there
  for all subsequent operations. On database commit, the in-memory index is
  persisted to disk.

  Attributes:
    db_path: The path to the database directory.
    db: The sqlite3 database connection.
    ui: The USearch index.
    _embedding_dim: The dimension of the embeddings.
    _embedding_dtype: The data type of the embeddings.
    _cursor: The sqlite3 cursor.
    _ui_loaded: Whether the USearch index was loaded in memory.
    _ui_updated: Whether the USearch index was updated since the last load and
      needs to be persisted to disk at some point in the future.
  """

  # User-provided.
  db_path: epath.Path

  # Instantiated during creation.
  db: sqlite3.Connection
  ui: uindex.Index

  # Obtained from `usearch_cfg`.
  _embedding_dim: int
  _embedding_dtype: type[Any] = np.float16

  # Dynamic state.
  _thread_local: threading.local = dataclasses.field(
      default_factory=threading.local
  )
  _ui_loaded: bool = False
  _ui_updated: bool = False
  _readonly: bool = False

  @property
  def sqlite_path(self) -> epath.Path:
    return self.db_path / HOPLITE_FILENAME

  @property
  def usearch_path(self) -> epath.Path:
    return self.db_path / UINDEX_FILENAME

  @staticmethod
  def _setup_tables(cursor: sqlite3.Cursor) -> None:
    """Create the SQLite tables.

    Args:
      cursor: The SQLite cursor to use.
    """

    # Skip setting up the tables if they already exist.
    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE name = "windows" AND type = "table"
        """)
    if cursor.fetchone() is not None:
      return

    # Enable foreign keys.
    cursor.execute('PRAGMA foreign_keys = ON')

    # Create the metadata table.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hoplite_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)

    # Create the deployments table.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            project TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            UNIQUE (name, project)
        )
        """)

    # Create the recordings table.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            datetime TEXT,
            deployment_id INTEGER REFERENCES deployments(id) ON DELETE CASCADE,
            UNIQUE (id, deployment_id),
            UNIQUE (filename, deployment_id)
        )
        """)

    # Create the windows table.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
            offsets FLOAT_LIST NOT NULL,
            UNIQUE (id, recording_id, offsets)
        )
        """)

    # Create the annotations table.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
            offsets FLOAT_LIST NOT NULL,
            label TEXT NOT NULL,
            label_type INTEGER NOT NULL,
            provenance TEXT NOT NULL,
            UNIQUE (id, recording_id, offsets)
        )
        """)

    # Create other indexes for efficient lookups.
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_annotations
        ON annotations(recording_id, offsets, label, label_type, provenance)
        """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_labels
        ON annotations(label, label_type, provenance)
        """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_recordings_deployment_id
        ON recordings(deployment_id)
        """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_windows_recording_id
        ON windows(recording_id)
        """)

  @staticmethod
  def _get_all_metadata(cursor: sqlite3.Cursor) -> config_dict.ConfigDict:
    """Get all key-value pairs from the metadata table.

    Args:
      cursor: The SQLite cursor to use.

    Returns:
      A ConfigDict containing all the metadata.
    """

    cursor.execute("""
        SELECT key, value
        FROM hoplite_metadata
        """)
    return config_dict.ConfigDict(
        {k: json.loads(v) for k, v in cursor.fetchall()}
    )

  @classmethod
  def create(  # pyrefly: ignore[bad-override]
      cls,
      db_path: str,
      usearch_cfg: config_dict.ConfigDict | None = None,
      readonly: bool = False,
  ) -> 'SQLiteUSearchDB':
    """Connect to and, if needed, initialize the database.

    Args:
      db_path: The path to the database directory.
      usearch_cfg: The configuration for the USearch index. If None, the config
        is loaded from the DB.
      readonly: If True, opens the database in read-only mode.

    Raises:
      ValueError: If `usearch_cfg` is inconsistent with the DB or if no config
        is found in the DB and none was provided.
      FileNotFoundError: If `readonly` is True and the USearch index file does
        not exist.

    Returns:
      A new instance of the database.
    """

    # Create the SQLite DB.
    db_path = epath.Path(db_path)  # pyrefly: ignore[bad-assignment]
    if not readonly:
      db_path.mkdir(parents=True, exist_ok=True)  # pyrefly: ignore[missing-attribute]
    sqlite_path = db_path / HOPLITE_FILENAME  # pyrefly: ignore[unsupported-operation]

    if readonly:
      db = sqlite3.connect(
          f'file:{sqlite_path.as_posix()}?mode=ro',
          uri=True,
          detect_types=sqlite3.PARSE_DECLTYPES,
          check_same_thread=False,
      )
    else:
      db = sqlite3.connect(
          sqlite_path.as_posix(),
          detect_types=sqlite3.PARSE_DECLTYPES,
          check_same_thread=False,
      )

    db.create_function(
        name='APPROX_FLOAT_LIST',
        narg=2,
        func=approx_float_list,
        deterministic=True,
    )
    db.create_function(
        name='GET_OFFSET_START',
        narg=1,
        func=get_offset_start,
        deterministic=True,
    )
    db.create_function(
        name='GET_OFFSET_END',
        narg=1,
        func=get_offset_end,
        deterministic=True,
    )
    db.set_trace_callback(
        lambda statement: logging.info('Executed SQL statement: %s', statement)
    )
    cursor = db.cursor()

    if not readonly:
      cursor.execute('PRAGMA journal_mode = WAL')  # Enable WAL mode.
      cls._setup_tables(cursor)
      db.commit()

    # Retrieve the metadata.
    # TODO(tomdenton): Check that `usearch_cfg` is consistent with the DB.
    metadata = cls._get_all_metadata(cursor)
    if (
        USEARCH_CONFIG_KEY in metadata
        and usearch_cfg is not None
        and metadata[USEARCH_CONFIG_KEY] != usearch_cfg
    ):
      raise ValueError(
          'A usearch_cfg was provided, but a different one already exists in'
          ' the DB.'
      )
    if USEARCH_CONFIG_KEY in metadata:
      usearch_cfg = metadata[USEARCH_CONFIG_KEY]
    elif usearch_cfg is None:
      raise ValueError('No usearch_cfg was found in DB and none was provided.')

    # Create the USearch index.
    usearch_dtype = USEARCH_DTYPES[usearch_cfg.dtype]
    index_path = db_path / UINDEX_FILENAME  # pyrefly: ignore[unsupported-operation]
    if index_path.exists():
      ui = uindex.Index(
          ndim=usearch_cfg.embedding_dim,
          path=index_path,
          view=True,
      )
      ui_in_memory = False
    else:
      if readonly:
        raise FileNotFoundError(f'USearch index not found at {index_path}')
      ui = uindex.Index(
          ndim=usearch_cfg.embedding_dim,
          metric=getattr(uindex.MetricKind, usearch_cfg.metric_name),
          expansion_add=usearch_cfg.expansion_add,
          expansion_search=usearch_cfg.expansion_search,
          dtype=usearch_dtype,
          path=index_path,
          view=False,
      )
      ui_in_memory = True

    # Create the Hoplite DB.
    hoplite_db = cls(
        db_path=db_path,  # pyrefly: ignore[bad-argument-type]
        db=db,
        ui=ui,
        _embedding_dim=usearch_cfg.embedding_dim,
        _embedding_dtype=usearch_cfg.dtype,
        _thread_local=threading.local(),
        _ui_loaded=ui_in_memory,
        _ui_updated=ui_in_memory,
        _readonly=readonly,
    )

    metadata_already_present = USEARCH_CONFIG_KEY in metadata
    if not readonly and not metadata_already_present:
      hoplite_db.insert_metadata(USEARCH_CONFIG_KEY, usearch_cfg)
      hoplite_db.commit()

    return hoplite_db

  def add_extra_table_column(
      self,
      table_name: str,
      column_name: str,
      column_type: type[Any],
  ) -> None:
    """Add an extra column to a table in the database."""

    if table_name not in [
        'deployments',
        'recordings',
        'windows',
        'annotations',
    ]:
      raise ValueError(f'Table `{table_name}` does not exist.')
    if not is_valid_sql_identifier(column_name):
      raise ValueError(f'Column `{column_name}` is not a valid SQL identifier.')
    if not isinstance(column_type, type):
      raise ValueError(f'Column type `{column_type}` must be a type.')

    if column_type not in PYTHON_TYPE_TO_SQL_TYPE:
      raise ValueError(
          f'Column type `{column_type.__name__}` is not supported. Use one of:'
          f' {", ".join([key.__name__ for key in PYTHON_TYPE_TO_SQL_TYPE.keys()])}'
      )

    cursor = self._get_cursor()
    cursor.execute(f'PRAGMA table_info({table_name})')
    existing_columns = {col_info[1] for col_info in cursor.fetchall()}
    if column_name in existing_columns:
      return

    cursor.execute(f"""
        ALTER TABLE {table_name}
        ADD COLUMN {column_name} {PYTHON_TYPE_TO_SQL_TYPE[column_type]}
        """)

    # Clear the cached property so that it is recomputed on the next access.
    self.__dict__.pop('_extra_table_columns', None)

  @functools.cached_property
  def _extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    """Get all extra columns in the database."""
    tables = ['deployments', 'recordings', 'windows', 'annotations']
    default_columns = {
        'deployments': {'id', 'name', 'project', 'latitude', 'longitude'},
        'recordings': {'id', 'filename', 'datetime', 'deployment_id'},
        'windows': {'id', 'recording_id', 'offsets'},
        'annotations': {
            'id',
            'recording_id',
            'offsets',
            'label',
            'label_type',
            'provenance',
        },
    }
    extra_columns = {t: {} for t in tables}
    cursor = self._get_cursor()
    for table in tables:
      cursor.execute(f'PRAGMA table_info({table})')
      columns_info = cursor.fetchall()
      for col_info in columns_info:
        # col_info: cid, name, type, notnull, dflt_value, pk
        col_name = col_info[1]
        col_type = col_info[2]
        if col_name not in default_columns[table]:
          try:
            extra_columns[table][col_name] = SQL_TYPE_TO_PYTHON_TYPE[col_type]
          except KeyError as e:
            raise ValueError(
                f'Unsupported column type {col_type} for column '
                f'{col_name} in table {table}'
            ) from e
    return extra_columns

  def get_extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    """Get all extra columns in the database."""
    return self._extra_table_columns

  def commit(self) -> None:
    """Commit any pending transactions to the database."""
    self.db.commit()
    if hasattr(self._thread_local, 'cursor') and self._thread_local.cursor is not None:
      self._thread_local.cursor.close()
      self._thread_local.cursor = None
    if self._ui_updated:
      self.ui.save()
      self._ui_updated = False

  def rollback(self) -> None:
    """Rollback any pending transactions to the database."""
    self.db.rollback()
    if hasattr(self._thread_local, 'cursor') and self._thread_local.cursor is not None:
      self._thread_local.cursor.close()
      self._thread_local.cursor = None

  def thread_split(self) -> 'SQLiteUSearchDB':
    """Get a new instance of the SQLite DB."""
    return self.create(self.db_path.as_posix(), readonly=self._readonly)

  def _get_cursor(self) -> sqlite3.Cursor:
    """Get a thread-local SQLite cursor.

    Each thread gets its own cursor to avoid issues with SQLite's
    thread-safety model. If the current thread doesn't have a cursor yet,
    one is created.

    Returns:
      A sqlite3.Cursor instance that is local to the current thread.
    """
    if not hasattr(self._thread_local, 'cursor') or self._thread_local.cursor is None:
      self._thread_local.cursor = self.db.cursor()
    return self._thread_local.cursor
