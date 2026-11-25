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
import json
import sqlite3
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
from perch_hoplite.db import sql_utils
from perch_hoplite.db import usearch_store
from usearch import index as uindex

HOPLITE_FILENAME = 'hoplite.sqlite'
UINDEX_FILENAME = usearch_store.UINDEX_FILENAME
USEARCH_CONFIG_KEY = usearch_store.USEARCH_CONFIG_KEY
DB_OPTIONS_KEY = usearch_store.DB_OPTIONS_KEY
USEARCH_DTYPES = usearch_store.USEARCH_DTYPES

# Re-export shared SQL and USearch helpers for backwards compatibility.
get_default_usearch_config = usearch_store.get_default_usearch_config
is_valid_sql_identifier = sql_utils.is_valid_sql_identifier
normalize_sql_value = sql_utils.normalize_sql_value
format_sql_insert_values = sql_utils.format_sql_insert_values
format_sql_update_on_conflict = sql_utils.format_sql_update_on_conflict
format_sql_where_conditions = sql_utils.format_sql_where_conditions
_get_window_query_components = sql_utils.get_window_query_components

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


@dataclasses.dataclass
class SQLiteUSearchDB(interface.HopliteDBInterface):
  """SQLite hoplite database, using optional USearch for vector storage.

  USearch provides both indexing for approximate nearest neighbor search and
  fast disk-based random access to vectors for the complete database. USearch
  is initialized lazily so databases created without embeddings (or before any
  embeddings are inserted) do not write an empty index file to disk. On
  database commit, any in-memory index modifications are persisted to disk.

  Attributes:
    db_path: The path to the database directory.
    db: The sqlite3 database connection.
    _vector_store: Optional USearch vector store wrapper.
    _cursor: The sqlite3 cursor.
    _readonly: Whether the database was opened in read-only mode.
  """

  # User-provided.
  db_path: epath.Path

  # Instantiated during creation.
  db: sqlite3.Connection
  _vector_store: usearch_store.USearchVectorStore

  # Dynamic state.
  _cursor: sqlite3.Cursor | None = None
  _readonly: bool = False

  @property
  def ui(self) -> uindex.Index:
    """The underlying USearch index (backwards-compatible property)."""
    return self._vector_store.ui

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
      use_embeddings: bool = True,
  ) -> 'SQLiteUSearchDB':
    """Connect to and, if needed, initialize the database.

    Args:
      db_path: The path to the database directory.
      usearch_cfg: The configuration for the USearch index. If None, the config
        is loaded from the DB (or embeddings are disabled if `use_embeddings` is
        False or the existing DB was created without embeddings).
      readonly: If True, opens the database in read-only mode.
      use_embeddings: Whether to enable USearch embedding storage. Pass `False`
        (with `usearch_cfg=None`) to create a database without a USearch index.
        When opening an existing database created with `use_embeddings=False`,
        this does not need to be repeated.

    Raises:
      ValueError: If `usearch_cfg` or `use_embeddings` is inconsistent with the
        DB, or if no config is found in the DB and none was provided.

    Returns:
      A new instance of the database.
    """
    resolved_db_path = epath.Path(db_path)
    sqlite_path = resolved_db_path / HOPLITE_FILENAME

    # Guard against accidentally creating an empty DB directory on a typo path
    # when the caller intended to open an existing embeddings DB.
    if (
        not readonly
        and not sqlite_path.exists()
        and use_embeddings
        and usearch_cfg is None
    ):
      raise ValueError(
          'No usearch_cfg was found in DB and none was provided. Pass'
          ' `usearch_cfg` to create an embeddings DB, or `use_embeddings=False`'
          ' to create a database without embeddings.'
      )

    if not readonly:
      resolved_db_path.mkdir(parents=True, exist_ok=True)

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

    metadata = cls._get_all_metadata(cursor)
    cursor.close()
    try:
      resolved_cfg, should_write_usearch_cfg, should_write_no_emb = (
          usearch_store.resolve_usearch_config(
              metadata=metadata,
              usearch_cfg=usearch_cfg,
              use_embeddings=use_embeddings,
              readonly=readonly,
          )
      )
      vector_store = usearch_store.USearchVectorStore(
          index_path=resolved_db_path / UINDEX_FILENAME,
          usearch_cfg=resolved_cfg,
          readonly=readonly,
      )
    except Exception:
      db.close()
      raise

    hoplite_db = cls(
        db_path=resolved_db_path,
        db=db,
        _vector_store=vector_store,
        _readonly=readonly,
    )

    if not readonly:
      if should_write_usearch_cfg and resolved_cfg is not None:
        hoplite_db.insert_metadata(USEARCH_CONFIG_KEY, resolved_cfg)
        if DB_OPTIONS_KEY in metadata:
          opts = config_dict.ConfigDict(metadata[DB_OPTIONS_KEY])
          opts.use_embeddings = True
          hoplite_db.insert_metadata(DB_OPTIONS_KEY, opts)
        hoplite_db.commit()
      elif should_write_no_emb:
        hoplite_db.insert_metadata(
            DB_OPTIONS_KEY,
            config_dict.ConfigDict({'use_embeddings': False}),
        )
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
    if self._cursor is not None:
      self._cursor.close()
      self._cursor = None
    self._vector_store.save()

  def rollback(self) -> None:
    """Rollback any pending transactions to the database."""
    self.db.rollback()
    if self._cursor is not None:
      self._cursor.close()
      self._cursor = None

  def close(self) -> None:
    """Close any underlying database resources."""
    if self._cursor is not None:
      self._cursor.close()
      self._cursor = None
    self.db.close()

  def thread_split(self) -> 'SQLiteUSearchDB':
    """Get a new instance of the SQLite DB."""
    return self.create(
        self.db_path.as_posix(),
        readonly=self._readonly,
        use_embeddings=self._vector_store.enabled,
    )

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

    for key, value in kwargs.items():
      if key not in self._extra_table_columns['deployments']:
        self.add_extra_table_column('deployments', key, type(value))
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

  def get_deployment(self, deployment_id: int) -> datatypes.Deployment:
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
    return datatypes.Deployment(**dict(zip(columns, result)))

  def remove_deployment(self, deployment_id: int) -> None:
    """Remove a deployment from the database."""

    deployment_id = int(deployment_id)

    remove_window_ids = self.match_window_ids(
        deployments_filter=config_dict.create(eq=dict(id=deployment_id))
    )
    if remove_window_ids:
      self._vector_store.remove(remove_window_ids)

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
    for key, value in kwargs.items():
      if key not in self._extra_table_columns['recordings']:
        self.add_extra_table_column('recordings', key, type(value))

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

  def get_recording(self, recording_id: int) -> datatypes.Recording:
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
    recording = datatypes.Recording(**dict(zip(columns, result)))
    if recording.datetime is not None:
      recording.datetime = dt.datetime.fromisoformat(recording.datetime)  # pyrefly: ignore[bad-argument-type]
    return recording

  def remove_recording(self, recording_id: int) -> None:
    """Remove a recording from the database."""

    recording_id = int(recording_id)

    remove_window_ids = self.match_window_ids(
        recordings_filter=config_dict.create(eq=dict(id=recording_id))
    )
    if remove_window_ids:
      self._vector_store.remove(remove_window_ids)

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
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
      **kwargs: Any,
  ) -> int:
    """Insert a window into the database."""

    if embedding is not None:
      self._vector_store.validate_embedding(embedding)

    duplicate_id = self._handle_window_duplicates(
        recording_id, offsets, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values(
        recording_id=recording_id,
        offsets=offsets,
        **kwargs,
    )
    try:
      cursor.execute(
          f"""
          INSERT INTO windows {columns_str}
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
          f'Error inserting the window into the database. {custom_msg}'
      ) from e

    window_id = cursor.lastrowid
    if window_id is None:
      raise RuntimeError('Error inserting the window into the database.')
    if embedding is not None:
      self._vector_store.add(window_id, embedding)
    return window_id

  def insert_windows_batch(
      self,
      windows_batch: Sequence[dict[str, Any]],
      embeddings_batch: np.ndarray | None = None,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
  ) -> Sequence[int]:
    """Insert a batch of windows into the database."""

    if embeddings_batch is not None:
      self._vector_store.validate_embedding(embeddings_batch)

    # Make sure that, unless we're in "allow" mode, there are no duplicates in
    # the batch itself.
    if handle_duplicates != 'allow':
      for i in range(len(windows_batch)):
        for j in range(i + 1, len(windows_batch)):
          if windows_batch[i]['recording_id'] == windows_batch[j][
              'recording_id'
          ] and np.allclose(
              windows_batch[i]['offsets'],
              windows_batch[j]['offsets'],
              rtol=0.0,
              atol=1e-6,
          ):
            raise RuntimeError(
                'Duplicates found in `windows_batch`, but this use case is not'
                ' supported unless `handle_duplicates` is set to "allow"'
                f' (handle_duplicates = "{handle_duplicates}").'
            )

    # Handle duplicates from the database.
    window_ids = [-1] * len(windows_batch)
    if handle_duplicates in ['overwrite', 'skip', 'error']:
      keep_idx = []
      remove_window_ids = set()

      # Go over all windows in the batch and check: (1) which ones need to be
      # removed, (2) which ones get a match and need to be skipped from
      # insertion, (3) which ones trigger an error, and (4) which ones still
      # need to be inserted.
      for idx, window_kwargs in enumerate(windows_batch):
        matches = self.get_all_windows(
            filter=config_dict.create(
                eq=dict(recording_id=window_kwargs['recording_id']),
                approx=dict(offsets=window_kwargs['offsets']),
            )
        )
        if matches:
          if handle_duplicates == 'overwrite':
            for match in matches:
              remove_window_ids.add(match.id)
            keep_idx.append(idx)
          elif handle_duplicates == 'skip':
            window_ids[idx] = matches[0].id
          elif handle_duplicates == 'error':
            raise RuntimeError(
                f'Duplicate window found (id = {matches[0].id}), but'
                ' `handle_duplicates` is set to "error".'
            )
        else:
          keep_idx.append(idx)

      # Now that no duplicates triggered an error, we can remove those that were
      # overwritten.
      for window_id in remove_window_ids:
        self.remove_window(window_id)

    else:
      keep_idx = list(range(len(windows_batch)))

    # Insert the windows that were not skipped and did not trigger an error.
    for idx in keep_idx:
      window_ids[idx] = self.insert_window(
          embedding=None,
          handle_duplicates='allow',
          **windows_batch[idx],
      )

    if embeddings_batch is not None and keep_idx:
      self._vector_store.add(
          np.array(window_ids)[keep_idx],
          embeddings_batch[keep_idx],
      )

    return window_ids

  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> datatypes.Window:
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
    window = datatypes.Window(
        embedding=None,
        **dict(zip(columns, result)),
    )
    if include_embedding:
      window.embedding = self.get_embedding(window_id)
    return window

  def get_window_annotations(
      self, window_id: int, label: str | None = None
  ) -> Sequence[datatypes.Annotation]:
    """Get all annotations intersecting the given window."""
    # Note that we may improve performance through memoization, though it would
    # need to be invalidated upon insertion/deletion of windows or annotations.
    window = self.get_window(window_id)
    w_start, w_end = window.offsets

    cursor = self._get_cursor()
    # Note that the inequalities should match the window.intersects() method.
    query = """
        SELECT *
        FROM annotations
        WHERE recording_id = ?
        AND GET_OFFSET_START(offsets) < ?
        AND GET_OFFSET_END(offsets) > ?
        """
    params = [window.recording_id, w_end, w_start]
    if label is not None:
      query += ' AND label = ?'
      params.append(label)  # pyrefly: ignore[bad-argument-type]
    cursor.execute(query, params)
    annotations = []
    columns = [col[0] for col in cursor.description]
    for result in cursor.fetchall():
      annotation = datatypes.Annotation(**dict(zip(columns, result)))
      annotation.label_type = datatypes.LabelType(annotation.label_type)
      annotations.append(annotation)
    return annotations

  def get_embedding(self, window_id: int) -> np.ndarray:
    """Get an embedding vector from the database."""
    return self._vector_store.get(int(window_id))

  def get_embeddings_batch(
      self,
      window_ids: Sequence[int],
  ) -> np.ndarray:
    """Get a batch of embedding vectors from the database."""
    return self._vector_store.get_batch(window_ids)

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

    self._vector_store.remove(window_id)

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

    duplicate_id = self._handle_annotation_duplicates(
        recording_id, offsets, label, label_type, provenance, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

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
    return cursor.lastrowid  # pyrefly: ignore[bad-return]

  def get_annotation(self, annotation_id: int) -> datatypes.Annotation:
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
    annotation = datatypes.Annotation(**dict(zip(columns, result)))
    annotation.label_type = datatypes.LabelType(annotation.label_type)
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
    return self._vector_store.count()

  def match_window_ids(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      annotations_filter: config_dict.ConfigDict | None = None,
      limit: int | None = None,
  ) -> Sequence[int]:
    """Get matching window IDs from the database based on given filters."""
    cursor = self._get_cursor()
    if annotations_filter:
      select_clause = 'SELECT DISTINCT windows.id'
    else:
      select_clause = 'SELECT windows.id'
    from_clause, where_clause, values = _get_window_query_components(
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        windows_filter=windows_filter,
        annotations_filter=annotations_filter,
    )

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
  ) -> Sequence[datatypes.Deployment]:
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
        datatypes.Deployment(**dict(zip(columns, result)))
        for result in cursor.fetchall()
    ]
    return deployments

  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Recording]:
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
      recording = datatypes.Recording(**dict(zip(columns, result)))
      if recording.datetime is not None:
        recording.datetime = dt.datetime.fromisoformat(recording.datetime)  # pyrefly: ignore[bad-argument-type]
      recordings.append(recording)
    return recordings

  def get_all_windows(
      self,
      include_embedding: bool = False,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
      annotations_filter: config_dict.ConfigDict | None = None,
  ) -> Sequence[datatypes.Window]:
    """Get all windows from the database."""
    cursor = self._get_cursor()
    if annotations_filter:
      select_clause = 'SELECT DISTINCT windows.*'
    else:
      select_clause = 'SELECT windows.*'
    from_clause, where_clause, values = _get_window_query_components(
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        windows_filter=filter,
        annotations_filter=annotations_filter,
    )

    # Execute the SQL query and return the window IDs.
    cursor.execute(
        f"""
        {select_clause}
        {from_clause}
        {where_clause}
        """,
        values,
    )

    windows = []
    columns = [col[0] for col in cursor.description]
    for result in cursor.fetchall():
      window = datatypes.Window(
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
  ) -> Sequence[datatypes.Annotation]:
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
      annotation = datatypes.Annotation(**dict(zip(columns, result)))
      annotation.label_type = datatypes.LabelType(annotation.label_type)
      annotations.append(annotation)
    return annotations

  def get_all_labels(
      self,
      label_type: datatypes.LabelType | None = None,
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
      label_type: datatypes.LabelType | None = None,
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
    return self._vector_store.embedding_dim

  def get_embedding_dtype(self) -> type[Any]:
    """Get the embedding data type."""
    return self._vector_store.embedding_dtype

  def search(
      self,
      query_embedding: np.ndarray,
      search_list_size: int,
      approximate: bool = True,
      target_score: float | None = None,
      score_fn_name: str = 'dot',
      **kwargs: Any,
  ) -> search_results.TopKSearchResults:
    """Search for neighbors of the query embedding.

    Args:
      query_embedding: The query embedding vector.
      search_list_size: The number of results to return.
      approximate: Whether to use approximate search.
      target_score: If set, search for examples near this score.
      score_fn_name: The name of the score function to use. Likely ignored if
        `approximate` is True, as the underlying indexed score function will be
        used.
      **kwargs: Additional keyword arguments to pass to the search function.

    Returns:
      A TopKSearchResults object containing the search results.
    """
    if not self._vector_store.enabled:
      raise ValueError('Embeddings are not enabled for this database.')
    if not approximate and target_score is not None:
      return super().search(
          query_embedding=query_embedding,
          search_list_size=search_list_size,
          approximate=approximate,
          target_score=target_score,
          score_fn_name=score_fn_name,
          **kwargs,
      )

    if target_score is not None:
      raise ValueError(
          'Approximate search does not support target_score sampling.'
      )

    results = search_results.TopKSearchResults(top_k=search_list_size)
    if self.count_embeddings() == 0:
      return results

    ann_matches = self._vector_store.search(
        query_embedding, count=search_list_size, exact=not approximate
    )
    for k, d in zip(ann_matches.keys, ann_matches.distances):
      # Usearch with IP returns 1 - inner_product.
      # We want higher is better, so 1 - d.
      results.update(search_results.SearchResult(k, 1.0 - d))
    score_fn = score_functions.get_score_fn(score_fn_name)
    results = brutalism.rerank(query_embedding, results, self, score_fn)
    return results
