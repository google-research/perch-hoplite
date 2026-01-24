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
from typing import Any

from absl import logging
from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import interface
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
      dtype=np.dtype('<f4'),  # little-endian np.float32
  ).tobytes()


def convert_float_list(blob: bytes) -> list[float]:
  return np.frombuffer(
      blob,
      dtype=np.dtype('<f4'),  # little-endian np.float32
  ).tolist()


def approx_float_list(blob: bytes, target: bytes) -> bool:
  return np.allclose(
      convert_float_list(blob),
      convert_float_list(target),
      rtol=0.0,
      atol=1e-6,
  )


sqlite3.register_adapter(list, adapt_float_list)
sqlite3.register_converter('FLOAT_LIST', convert_float_list)


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

  if isinstance(value, list) or isinstance(value, tuple):
    return [normalize_sql_value(v) for v in value]
  if isinstance(value, interface.LabelType):
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
  _cursor: sqlite3.Cursor | None = None
  _ui_loaded: bool = False
  _ui_updated: bool = False

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
            UNIQUE (id, recording_id, offsets),
            UNIQUE (recording_id, offsets)
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
  def create(
      cls,
      db_path: str,
      usearch_cfg: config_dict.ConfigDict | None = None,
  ) -> 'SQLiteUSearchDB':
    """Connect to and, if needed, initialize the database.

    Args:
      db_path: The path to the database directory.
      usearch_cfg: The configuration for the USearch index. If None, the config
        is loaded from the DB.

    Returns:
      A new instance of the database.
    """

    # Create the SQLite DB.
    db_path = epath.Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    sqlite_path = db_path / HOPLITE_FILENAME
    db = sqlite3.connect(
        sqlite_path.as_posix(),
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    db.create_function(
        name='APPROX_FLOAT_LIST',
        narg=2,
        func=approx_float_list,
        deterministic=True,
    )
    db.set_trace_callback(
        lambda statement: logging.info('Executed SQL statement: %s', statement)
    )
    cursor = db.cursor()
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
    index_path = db_path / UINDEX_FILENAME
    if index_path.exists():
      ui = uindex.Index(
          ndim=usearch_cfg.embedding_dim,
          path=index_path,
          view=True,
      )
      ui_in_memory = False
    else:
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
        db_path=db_path,
        db=db,
        ui=ui,
        _embedding_dim=usearch_cfg.embedding_dim,
        _embedding_dtype=usearch_cfg.dtype,
        _ui_loaded=ui_in_memory,
        _ui_updated=ui_in_memory,
    )
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
    if self._cursor is not None:
      self._cursor.close()
      self._cursor = None
    if self._ui_updated:
      self.ui.save()
      self._ui_updated = False

  def rollback(self) -> None:
    """Rollback any pending transactions to the database."""
    self.db.rollback()
    if self._cursor is not None:
      self._cursor.close()
      self._cursor = None

  def thread_split(self) -> 'SQLiteUSearchDB':
    """Get a new instance of the SQLite DB."""
    return self.create(self.db_path.as_posix())

  def _get_cursor(self) -> sqlite3.Cursor:
    """Get the SQLite cursor."""
    if self._cursor is None:
      self._cursor = self.db.cursor()
    return self._cursor

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    """Insert a key-value pair into the metadata table."""

    cursor = self._get_cursor()
    json_coded = value.to_json()
    cursor.execute(
        """
        INSERT INTO hoplite_metadata (key, value)
        VALUES (?, ?)
        ON CONFLICT (key)
        DO UPDATE SET value = excluded.value
        """,
        (key, json_coded),
    )

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    """Get a key-value pair from the metadata table."""

    cursor = self._get_cursor()

    if key is None:
      cursor.execute("""
          SELECT key, value
          FROM hoplite_metadata
          """)
      return config_dict.ConfigDict(
          {k: json.loads(v) for k, v in cursor.fetchall()}
      )

    cursor.execute(
        """
        SELECT value
        FROM hoplite_metadata
        WHERE key = ?
        """,
        (key,),
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Metadata key not found: {key}')
    return config_dict.ConfigDict(json.loads(result[0]))

  def remove_metadata(self, key: str | None) -> None:
    """Remove a key-value pair from the metadata table."""

    cursor = self._get_cursor()

    if key is None:
      cursor.execute("""
          DELETE FROM hoplite_metadata
          """)
      return

    cursor.execute(
        """
        DELETE FROM hoplite_metadata
        WHERE key = ?
        """,
        (key,),
    )
    if cursor.rowcount == 0:
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

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values(
        name=name,
        project=project,
        latitude=latitude,
        longitude=longitude,
        **kwargs,
    )
    update_clause_str = format_sql_update_on_conflict(
        'latitude',
        'longitude',
        *self._extra_table_columns['deployments'].keys(),
    )
    cursor.execute(
        f"""
        INSERT INTO deployments {columns_str}
        VALUES {placeholders_str}
        ON CONFLICT (name, project)
        {update_clause_str}
        """,
        values,
    )

    deployment_id = cursor.lastrowid
    if deployment_id is None:
      raise RuntimeError('Error inserting the deployment into the database.')
    return deployment_id

  def get_deployment(self, deployment_id: int) -> interface.Deployment:
    """Get a deployment from the database."""

    deployment_id = int(deployment_id)

    cursor = self._get_cursor()
    cursor.execute(
        """
        SELECT *
        FROM deployments
        WHERE id = ?
        """,
        (deployment_id,),
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Deployment id not found: {deployment_id}')

    columns = [col[0] for col in cursor.description]
    return interface.Deployment(**dict(zip(columns, result)))

  def remove_deployment(self, deployment_id: int) -> None:
    """Remove a deployment from the database."""

    deployment_id = int(deployment_id)

    remove_window_ids = self.match_window_ids(
        deployments_filter=config_dict.create(
            eq=dict(deployment_id=deployment_id)
        )
    )
    if remove_window_ids:
      if not self._ui_loaded:
        self.ui.load()
        self._ui_loaded = True
      self.ui.remove(remove_window_ids)
      self._ui_updated = True

    cursor = self._get_cursor()
    cursor.execute(
        """
        DELETE FROM deployments
        WHERE id = ?
        """,
        (deployment_id,),
    )
    if cursor.rowcount == 0:
      raise KeyError(f'Deployment id not found: {deployment_id}')

  def insert_recording(
      self,
      filename: str,
      datetime: dt.datetime | None = None,
      deployment_id: int | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a recording into the database."""

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values(
        filename=filename,
        datetime=datetime,
        deployment_id=deployment_id,
        **kwargs,
    )
    update_clause_str = format_sql_update_on_conflict(
        'datetime',
        *self._extra_table_columns['recordings'].keys(),
    )
    try:
      cursor.execute(
          f"""
          INSERT INTO recordings {columns_str}
          VALUES {placeholders_str}
          ON CONFLICT (filename, deployment_id)
          {update_clause_str}
          """,
          values,
      )
    except sqlite3.Error as e:
      if e.sqlite_errorname == 'SQLITE_CONSTRAINT_FOREIGNKEY':
        custom_msg = 'Check that the deployment_id exists.'
      else:
        custom_msg = ''
      raise RuntimeError(
          f'Error inserting the recording into the database. {custom_msg}'
      ) from e

    recording_id = cursor.lastrowid
    if recording_id is None:
      raise RuntimeError('Error inserting the recording into the database.')
    return recording_id

  def get_recording(self, recording_id: int) -> interface.Recording:
    """Get a recording from the database."""

    recording_id = int(recording_id)

    cursor = self._get_cursor()
    cursor.execute(
        """
        SELECT *
        FROM recordings
        WHERE id = ?
        """,
        (recording_id,),
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Recording id not found: {recording_id}')

    columns = [col[0] for col in cursor.description]
    recording = interface.Recording(**dict(zip(columns, result)))
    if recording.datetime is not None:
      recording.datetime = dt.datetime.fromisoformat(recording.datetime)
    return recording

  def remove_recording(self, recording_id: int) -> None:
    """Remove a recording from the database."""

    recording_id = int(recording_id)

    remove_window_ids = self.match_window_ids(
        recordings_filter=config_dict.create(eq=dict(recording_id=recording_id))
    )
    if remove_window_ids:
      if not self._ui_loaded:
        self.ui.load()
        self._ui_loaded = True
      self.ui.remove(remove_window_ids)
      self._ui_updated = True

    cursor = self._get_cursor()
    cursor.execute(
        """
        DELETE FROM recordings
        WHERE id = ?
        """,
        (recording_id,),
    )
    if cursor.rowcount == 0:
      raise KeyError(f'Recording id not found: {recording_id}')

  def insert_window(
      self,
      recording_id: int,
      offsets: list[float],
      embedding: np.ndarray | None = None,
      **kwargs: Any,
  ) -> int:
    """Insert a window into the database."""

    if embedding is not None and embedding.shape[-1] != self._embedding_dim:
      raise ValueError(
          f'Incorrect embedding dimension. Expected {self._embedding_dim}, but'
          f' got {embedding.shape[-1]}.'
      )

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values(
        recording_id=recording_id,
        offsets=offsets,
        **kwargs,
    )
    update_clause_str = format_sql_update_on_conflict(
        *self._extra_table_columns['windows'].keys(),
    )
    try:
      cursor.execute(
          f"""
          INSERT INTO windows {columns_str}
          VALUES {placeholders_str}
          ON CONFLICT (recording_id, offsets)
          {update_clause_str}
          """,
          values,
      )
    except sqlite3.Error as e:
      if e.sqlite_errorname == 'SQLITE_CONSTRAINT_FOREIGNKEY':
        custom_msg = 'Check that the recording_id exists.'
      else:
        custom_msg = ''
      raise RuntimeError(
          f'Error inserting the window into the database. {custom_msg}'
      ) from e

    window_id = cursor.lastrowid
    if window_id is None:
      raise RuntimeError('Error inserting the window into the database.')
    if embedding is not None:
      if not self._ui_loaded:
        self.ui.load()
        self._ui_loaded = True
      self.ui.add(window_id, embedding.astype(self._embedding_dtype))
      self._ui_updated = True
    return window_id

  def insert_windows_batch(
      self,
      windows_batch: Sequence[dict[str, Any]],
      embeddings_batch: np.ndarray | None = None,
  ) -> Sequence[int]:
    """Insert a batch of windows into the database."""

    if (
        embeddings_batch is not None
        and embeddings_batch.shape[-1] != self._embedding_dim
    ):
      raise ValueError(
          f'Incorrect embedding dimension. Expected {self._embedding_dim}, but'
          f' got {embeddings_batch.shape[-1]}.'
      )

    window_ids = [
        self.insert_window(embedding=None, **window_kwargs)
        for window_kwargs in windows_batch
    ]

    if embeddings_batch is not None:
      if not self._ui_loaded:
        self.ui.load()
        self._ui_loaded = True
      self.ui.add(window_ids, embeddings_batch.astype(self._embedding_dtype))
      self._ui_updated = True

    return window_ids

  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> interface.Window:
    """Get a window from the database."""

    window_id = int(window_id)

    cursor = self._get_cursor()
    cursor.execute(
        """
        SELECT *
        FROM windows
        WHERE id = ?
        """,
        (window_id,),
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Window id not found: {window_id}')

    columns = [col[0] for col in cursor.description]
    window = interface.Window(
        embedding=None,
        **dict(zip(columns, result)),
    )
    if include_embedding:
      window.embedding = self.get_embedding(window_id)
    return window

  def get_embedding(self, window_id: int) -> np.ndarray:
    """Get an embedding vector from the database."""

    window_id = int(window_id)

    found = self.ui.contains(window_id)
    if not isinstance(found, bool):
      raise RuntimeError(
          'Expected bool result from the USearch `contains()` method, but got'
          f' {type(found)}: {found}.'
      )
    if not found:
      raise KeyError(f'Embedding vector not found for window id: {window_id}')
    embedding = self.ui.get(window_id)
    if not isinstance(embedding, np.ndarray):
      raise RuntimeError(
          'Expected np.ndarray result from the USearch `get()` method, but got'
          f' {type(embedding)}: {embedding}.'
      )
    return embedding

  def get_embeddings_batch(
      self,
      window_ids: Sequence[int],
  ) -> np.ndarray:
    """Get a batch of embedding vectors from the database."""

    found = self.ui.contains(window_ids)
    if not isinstance(found, np.ndarray):
      raise RuntimeError(
          'Expected np.ndarray result from the USearch `contains()` method, but'
          f' got {type(found)}: {found}.'
      )
    if not np.all(found):
      raise KeyError(
          'Embedding vectors not found for window ids:'
          f' {itertools.compress(window_ids, ~found)}'
      )
    embeddings_batch = self.ui.get(window_ids)
    if not isinstance(embeddings_batch, np.ndarray):
      raise RuntimeError(
          'Expected np.ndarray result from the USearch `get()` method, but got'
          f' {type(embeddings_batch)}: {embeddings_batch}.'
      )
    return embeddings_batch

  def remove_window(self, window_id: int) -> None:
    """Remove a window from the database."""

    window_id = int(window_id)

    cursor = self._get_cursor()
    cursor.execute(
        """
        DELETE FROM windows
        WHERE id = ?
        """,
        (window_id,),
    )
    if cursor.rowcount == 0:
      raise KeyError(f'Window id not found: {window_id}')

    if not self._ui_loaded:
      self.ui.load()
      self._ui_loaded = True
    self.ui.remove(window_id)
    self._ui_updated = True

  def insert_annotation(
      self,
      recording_id: int,
      offsets: list[float],
      label: str,
      label_type: interface.LabelType,
      provenance: str,
      skip_duplicates: bool = False,
      **kwargs: Any,
  ) -> int:
    """Insert an annotation into the database."""

    if skip_duplicates:
      matches = self.get_all_annotations(
          filter=config_dict.create(
              eq=dict(
                  recording_id=recording_id, label=label, label_type=label_type
              ),
              approx=dict(offsets=offsets),
          )
      )
      if matches:
        return matches[0].id

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values(
        recording_id=recording_id,
        offsets=offsets,
        label=label,
        label_type=label_type,
        provenance=provenance,
        **kwargs,
    )
    try:
      cursor.execute(
          f"""
          INSERT INTO annotations {columns_str}
          VALUES {placeholders_str}
          """,
          values,
      )
    except sqlite3.Error as e:
      if e.sqlite_errorname == 'SQLITE_CONSTRAINT_FOREIGNKEY':
        custom_msg = 'Check that the recording_id exists.'
      else:
        custom_msg = ''
      raise RuntimeError(
          f'Error inserting the annotation into the database. {custom_msg}'
      ) from e

    annotation_id = cursor.lastrowid
    if annotation_id is None:
      raise RuntimeError('Error inserting the annotation into the database.')
    return cursor.lastrowid

  def get_annotation(self, annotation_id: int) -> interface.Annotation:
    """Get an annotation from the database."""

    annotation_id = int(annotation_id)

    cursor = self._get_cursor()
    cursor.execute(
        """
        SELECT *
        FROM annotations
        WHERE id = ?
        """,
        (annotation_id,),
    )
    result = cursor.fetchone()
    if result is None:
      raise KeyError(f'Annotation id not found: {annotation_id}')

    columns = [col[0] for col in cursor.description]
    annotation = interface.Annotation(**dict(zip(columns, result)))
    annotation.label_type = interface.LabelType(annotation.label_type)
    return annotation

  def remove_annotation(self, annotation_id: int) -> None:
    """Remove an annotation from the database."""

    annotation_id = int(annotation_id)

    cursor = self._get_cursor()
    cursor.execute(
        """
        DELETE FROM annotations
        WHERE id = ?
        """,
        (annotation_id,),
    )
    if cursor.rowcount == 0:
      raise KeyError(f'Annotation id not found: {annotation_id}')

  def count_embeddings(self) -> int:
    """Get the number of embeddings in the database."""
    return self.ui.size

  def match_window_ids(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      annotations_filter: config_dict.ConfigDict | None = None,
      limit: int | None = None,
  ) -> Sequence[int]:
    """Get matching window IDs from the database based on given filters."""

    # TODO(stefanistrate): Is it more efficient for large databases to run
    # multiple SELECT queries (in order: by deployment, by recording, by window,
    # by annotation) instead of running one giant JOIN?

    cursor = self._get_cursor()

    # Pick which tables need to be queried.
    query_tables = {'windows'}
    if annotations_filter:
      query_tables |= {'annotations'}
    if recordings_filter:
      query_tables |= {'recordings'}
    if deployments_filter:
      query_tables |= {'recordings', 'deployments'}

    # Build the `SELECT ... FROM ... [JOIN ...]` part of the SQL query.
    if 'annotations' in query_tables:
      select_clause = 'SELECT DISTINCT windows.id'
    else:
      select_clause = 'SELECT windows.id'
    from_clause = 'FROM windows'
    if 'annotations' in query_tables:
      from_clause += (
          ' JOIN annotations ON windows.recording_id = annotations.recording_id'
          ' AND APPROX_FLOAT_LIST(windows.offsets, annotations.offsets) = TRUE'
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
                annotations_filter, table_prefix='annotations'
            ),
            format_sql_where_conditions(windows_filter, table_prefix='windows'),
            format_sql_where_conditions(
                recordings_filter, table_prefix='recordings'
            ),
            format_sql_where_conditions(
                deployments_filter, table_prefix='deployments'
            ),
        ])
    )
    conditions_str = ' AND '.join(c for c in conditions if c)
    values = list(itertools.chain.from_iterable(values))
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''

    # Build the `LIMIT ...` part of the SQL query.
    if limit is None:
      limit_clause = ''
    else:
      limit_clause = f'LIMIT {limit}'

    # Execute the SQL query and return the window IDs.
    cursor.execute(
        f"""
        {select_clause}
        {from_clause}
        {where_clause}
        {limit_clause}
        """,
        values,
    )
    return [result[0] for result in cursor.fetchall()]

  def get_all_projects(self) -> Sequence[str]:
    """Get all distinct projects from the database."""

    cursor = self._get_cursor()
    cursor.execute("""
        SELECT DISTINCT project
        FROM deployments
        ORDER BY project
        """)
    return [result[0] for result in cursor.fetchall()]

  def get_all_deployments(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[interface.Deployment]:
    """Get all deployments from the database."""

    cursor = self._get_cursor()
    conditions_str, values = format_sql_where_conditions(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(
        f"""
        SELECT *
        FROM deployments
        {where_clause}
        """,
        values,
    )

    columns = [col[0] for col in cursor.description]
    deployments = [
        interface.Deployment(**dict(zip(columns, result)))
        for result in cursor.fetchall()
    ]
    return deployments

  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[interface.Recording]:
    """Get all recordings from the database."""

    cursor = self._get_cursor()
    conditions_str, values = format_sql_where_conditions(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(
        f"""
        SELECT *
        FROM recordings
        {where_clause}
        """,
        values,
    )

    recordings = []
    columns = [col[0] for col in cursor.description]
    for result in cursor.fetchall():
      recording = interface.Recording(**dict(zip(columns, result)))
      if recording.datetime is not None:
        recording.datetime = dt.datetime.fromisoformat(recording.datetime)
      recordings.append(recording)
    return recordings

  def get_all_windows(
      self,
      include_embedding: bool = False,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[interface.Window]:
    """Get all windows from the database."""

    cursor = self._get_cursor()
    conditions_str, values = format_sql_where_conditions(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(
        f"""
        SELECT *
        FROM windows
        {where_clause}
        """,
        values,
    )

    windows = []
    columns = [col[0] for col in cursor.description]
    for result in cursor.fetchall():
      window = interface.Window(
          embedding=None,
          **dict(zip(columns, result)),
      )
      if include_embedding:
        window.embedding = self.get_embedding(window.id)
      windows.append(window)
    return windows

  def get_all_annotations(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[interface.Annotation]:
    """Get all annotations from the database."""

    cursor = self._get_cursor()
    conditions_str, values = format_sql_where_conditions(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(
        f"""
        SELECT *
        FROM annotations
        {where_clause}
        """,
        values,
    )

    annotations = []
    columns = [col[0] for col in cursor.description]
    for result in cursor.fetchall():
      annotation = interface.Annotation(**dict(zip(columns, result)))
      annotation.label_type = interface.LabelType(annotation.label_type)
      annotations.append(annotation)
    return annotations

  def get_all_labels(
      self,
      label_type: interface.LabelType | None = None,
  ) -> Sequence[str]:
    """Get all distinct labels from the database."""

    cursor = self._get_cursor()
    if label_type is None:
      where_clause = ''
      values = tuple()
    else:
      filter_dict = config_dict.create(eq=dict(label_type=label_type))
      conditions_str, values = format_sql_where_conditions(filter_dict)
      where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(
        f"""
        SELECT DISTINCT label
        FROM annotations
        {where_clause}
        ORDER BY label
        """,
        values,
    )
    return [result[0] for result in cursor.fetchall()]

  def count_each_label(
      self,
      label_type: interface.LabelType | None = None,
  ) -> collections.Counter[str]:
    """Count each label in the database, ignoring provenance."""

    cursor = self._get_cursor()
    if label_type is None:
      where_clause = ''
      values = tuple()
    else:
      filter_dict = config_dict.create(eq=dict(label_type=label_type))
      conditions_str, values = format_sql_where_conditions(filter_dict)
      where_clause = f'WHERE {conditions_str}' if conditions_str else ''

    # Subselect with DISTINCT is needed to avoid double-counting the same label
    # on the same recording offsets because of different provenances.
    cursor.execute(
        f"""
        SELECT label, COUNT(*)
        FROM (
            SELECT DISTINCT recording_id, offsets, label, label_type
            FROM annotations
            {where_clause}
        )
        GROUP BY label
        ORDER BY label
        """,
        values,
    )
    return collections.Counter(
        {result[0]: result[1] for result in cursor.fetchall()}
    )

  def get_embedding_dim(self) -> int:
    """Get the embedding dimension."""
    return self._embedding_dim

  def get_embedding_dtype(self) -> type[Any]:
    """Get the embedding data type."""
    return self._embedding_dtype
