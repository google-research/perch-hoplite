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

"""Database of bioacoustic label domains.

This module provides the TaxonomyDatabase class, which handles namespaces of
class labels, mapped pairs, and class lists using an SQLite backend.

Database Schema Design:
1. strings: A unique registry of all text values (labels, namespace names,
   etc.) loaded into the database, mapped to auto-incrementing integer uids to
   avoid duplicating strings.
2. namespaces: Stores basic namespace entities, mapping names to a set of class
   label string uids. To avoid duplicate string entries and keep space footprint
   small, the class uids are serialized into a compressed BLOB.
3. class_lists: Stores class list metadata (name, namespace_name) and their
   ordered sequence of classes, serialized as a compressed BLOB of label
   string uids.
4. mappings: Stores conversion mappings from a source namespace to a target
   namespace, along with is_default flag and parallel source/target uid lists
   (as compressed BLOBs) representing override mapped pairs (excluding
   identity mappings).

Algebraic Operations & Notation:
- Set Arithmetic: The Namespace class implements set union (+) and set
  difference (-).
- Notation Parser: A recursive-descent parser supports evaluating dynamic,
  nested algebraic expressions (e.g. db.namespaces['(A + B) - C']), which
  resolves the base namespaces from SQLite and computes the resulting Namespace
  on the fly.

Extended Identity:
- Mappings have a default boolean flag. If True (or if evaluated as a default
  mapping), any shared items x in the intersection of the source and target
  namespaces will map to themselves by default (m(x) = x) without needing to
  be explicitly recorded as mappings. Explicit entries in mappings (stored
  as source_uids and target_uids BLOBs) override this behavior.
"""

import collections.abc
import functools
import json
import os
import re
import sqlite3
import typing
import zlib

import numpy as np
from perch_hoplite import path_utils
from perch_hoplite.taxonomy import namespace

TAXONOMY_DATABASE_FILENAME = "taxonomy/taxonomy_database.sqlite"


ClassListType = str | namespace.ClassList | tuple[str, ...]
MappingType = str | namespace.Mapping | dict[str, str]


def get_classes(class_list: ClassListType) -> tuple[str, ...]:
  """Load classes from the namespace database.

  Args:
    class_list: Name of the class list to load. This can be a class list,
      namespace, or mapping name. If it is a mapping name, the sorted tuple of
      all target classes is returned. If an actual ClassList is passed, the
      tuple of classes is returned.

  Returns:
    A tuple of classes.
  """
  if isinstance(class_list, namespace.ClassList):
    return class_list.classes
  elif isinstance(class_list, tuple):
    return class_list

  db = load_db()
  if class_list in db.class_lists:
    return db.class_lists[class_list].classes
  elif class_list in db.namespaces:
    return tuple(sorted(tuple(db.namespaces[class_list].classes)))
  elif class_list in db.mappings:
    image_classes = db.mappings[class_list].mapped_pairs.values()
    return tuple(sorted(tuple(image_classes)))
  else:
    raise ValueError(
        "Class list %s not found in namespace database." % class_list
    )


def get_mapping(mapping: MappingType) -> dict[str, str]:
  """Load mapping from the namespace database."""
  if isinstance(mapping, namespace.Mapping):
    return mapping.mapped_pairs
  elif isinstance(mapping, dict):
    return mapping
  db = load_db()
  if mapping in db.mappings:
    return db.mappings[mapping].mapped_pairs
  else:
    raise ValueError("Mapping %s not found in namespace database." % mapping)


def num_classes(class_list: ClassListType) -> int:
  """Return the number of classes in the class list."""
  return len(get_classes(class_list))


def create_tables(conn: sqlite3.Connection) -> None:
  """Creates all necessary SQLite tables and indexes."""
  cursor = conn.cursor()
  cursor.execute("PRAGMA foreign_keys = ON;")

  cursor.execute("""
  CREATE TABLE IF NOT EXISTS strings (
      uid INTEGER PRIMARY KEY AUTOINCREMENT,
      value TEXT
  );
  """)

  cursor.execute("""
  CREATE TABLE IF NOT EXISTS namespaces (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT UNIQUE,
      classes_uids BLOB
  );
  """)

  cursor.execute("""
  CREATE TABLE IF NOT EXISTS class_lists (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT UNIQUE,
      namespace_name TEXT,
      classes_uids BLOB
  );
  """)

  cursor.execute("""
  CREATE TABLE IF NOT EXISTS mappings (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT UNIQUE,
      source_namespace_name TEXT,
      target_namespace_name TEXT,
      is_default INTEGER DEFAULT 0,
      source_uids BLOB,
      target_uids BLOB
  );
  """)
  conn.commit()


def get_or_insert_strings(
    conn: sqlite3.Connection, values: list[str]
) -> dict[str, int]:
  """Ensures all strings in values are recorded, returning their uids."""
  if not values:
    return {}
  cursor = conn.cursor()
  cursor.execute(
      "CREATE UNIQUE INDEX IF NOT EXISTS idx_strings_value ON strings(value);"
  )
  unique_values = list(set(values))

  chunk_size = 500
  existing = {}
  for i in range(0, len(unique_values), chunk_size):
    chunk = unique_values[i : i + chunk_size]
    placeholders = ",".join("?" for _ in chunk)
    cursor.execute(
        f"SELECT uid, value FROM strings WHERE value IN ({placeholders})", chunk
    )
    for uid, val in cursor.fetchall():
      existing[val] = uid

  to_insert = [val for val in unique_values if val not in existing]
  if to_insert:
    cursor.executemany(
        "INSERT OR IGNORE INTO strings (value) VALUES (?)",
        [(val,) for val in to_insert],
    )
    for i in range(0, len(to_insert), chunk_size):
      chunk = to_insert[i : i + chunk_size]
      placeholders = ",".join("?" for _ in chunk)
      cursor.execute(
          f"SELECT uid, value FROM strings WHERE value IN ({placeholders})",
          chunk,
      )
      for uid, val in cursor.fetchall():
        existing[val] = uid
  return existing


def parse_namespace_expression(
    expression: str,
    get_base_namespace_fn: typing.Callable[[str], typing.AbstractSet[str]],
) -> set[str]:
  """Parses algebraic namespace expressions like (A + B) - C."""
  tokens = re.findall(r"[A-Za-z0-9_]+|[-+()]", expression)
  pos = 0

  def peek():
    nonlocal pos
    if pos < len(tokens):
      return tokens[pos]
    return None

  def consume(expected=None):
    nonlocal pos
    t = peek()
    if t is None:
      raise ValueError(
          f"Parse error in namespace expression '{expression}': unexpected end"
          " of input"
      )
    if expected is not None and t != expected:
      raise ValueError(
          f"Parse error in namespace expression '{expression}': expected"
          f" '{expected}', got '{t}'"
      )
    pos += 1
    return t

  def parse_expression() -> set[str]:
    result = parse_term()
    while True:
      op = peek()
      if op in ("+", "-"):
        consume()
        right = parse_term()
        if op == "+":
          result = result.union(right)
        else:
          result = result.difference(right)
      else:
        break
    return result

  def parse_term() -> set[str]:
    t = peek()
    if t == "(":
      consume("(")
      result = parse_expression()
      consume(")")
      return result
    elif t is not None and re.match(r"^[A-Za-z0-9_]+$", t):
      name = consume()
      return set(get_base_namespace_fn(name))
    else:
      raise ValueError(
          f"Parse error in namespace expression '{expression}': unexpected"
          f" token '{t}'"
      )

  result = parse_expression()
  if pos < len(tokens):
    raise ValueError(
        f"Parse error in namespace expression '{expression}': unexpected"
        f" trailing tokens: {tokens[pos:]}"
    )
  return result


def serialize_uids(uids: typing.Any) -> bytes:
  """Serializes a numpy array of UIDs to bytes with dtype compression and zlib."""
  uids = np.asarray(uids)
  if uids.size == 0:
    return b""
  max_val = np.max(uids)
  if max_val < 256:
    dtype = np.uint8
    label = 0
  elif max_val < 65536:
    dtype = np.uint16
    label = 1
  else:
    dtype = np.uint32
    label = 2
  raw_bytes = bytes([label]) + uids.astype(dtype).tobytes()
  return zlib.compress(raw_bytes)


def deserialize_uids(blob: bytes, reshape_to_2d: bool = False) -> np.ndarray:
  """Deserializes UIDs from bytes, detecting the correct dtype."""
  if not blob:
    return np.array([], dtype=np.int32)
  decompressed = zlib.decompress(blob)
  label = decompressed[0]
  if label == 0:
    dtype = np.uint8
  elif label == 1:
    dtype = np.uint16
  else:
    dtype = np.uint32
  uids = np.frombuffer(decompressed[1:], dtype=dtype)
  if reshape_to_2d:
    return uids.reshape(2, -1)
  return uids


class SQLiteNamespacesDict(collections.abc.MutableMapping):
  """Dictionary-like proxy for namespaces in SQLite."""

  def __init__(self, db: "TaxonomyDatabase"):
    self._db = db

  def __ior__(self, other: typing.Any) -> typing.Any:
    self.update(other)
    return self

  def __or__(self, other: typing.Any) -> typing.Any:
    res = dict(self)
    res.update(other)
    return res

  def __getitem__(self, key: str) -> namespace.Namespace:
    if "+" in key or "-" in key:
      try:
        classes = parse_namespace_expression(
            key, lambda name: self[name].classes
        )
        return namespace.Namespace(classes=frozenset(classes))
      except (ValueError, KeyError) as e:
        raise KeyError(
            f"Could not resolve algebraic namespace '{key}': {e}"
        ) from e

    cursor = self._db.conn.cursor()
    cursor.execute("SELECT classes_uids FROM namespaces WHERE name = ?", (key,))
    row = cursor.fetchone()
    if row is None:
      raise KeyError(key)
    classes_uids_blob = row[0]
    if not classes_uids_blob:
      return namespace.Namespace(classes=frozenset())

    uids = deserialize_uids(classes_uids_blob)
    if len(uids) < 1000:
      placeholders = ",".join("?" for _ in uids)
      cursor.execute(
          f"SELECT value FROM strings WHERE uid IN ({placeholders})",
          [int(x) for x in uids],
      )
      classes = [r[0] for r in cursor.fetchall()]
    else:
      uid_to_str = self._db.get_uid_to_str()
      classes = [uid_to_str[uid] for uid in uids if uid in uid_to_str]
    return namespace.Namespace(classes=frozenset(classes))

  def __setitem__(self, key: str, value: namespace.Namespace) -> None:
    if not isinstance(value, namespace.Namespace):
      raise TypeError("Value must be a Namespace object")
    if "+" in key or "-" in key:
      raise ValueError("Cannot write to an algebraic namespace name")

    cursor = self._db.conn.cursor()
    classes = list(value.classes)
    uids_dict = get_or_insert_strings(self._db.conn, classes)
    uids = np.array([uids_dict[cls] for cls in classes], dtype=np.int32)
    classes_uids_blob = serialize_uids(uids)

    cursor.execute("DELETE FROM namespaces WHERE name = ?", (key,))
    cursor.execute(
        "INSERT INTO namespaces (name, classes_uids) VALUES (?, ?)",
        (key, classes_uids_blob),
    )
    self._db.clear_string_caches()
    self._db.conn.commit()

  def __delitem__(self, key: str) -> None:
    if "+" in key or "-" in key:
      raise ValueError("Cannot delete an algebraic namespace name")
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT 1 FROM namespaces WHERE name = ?", (key,))
    if cursor.fetchone() is None:
      raise KeyError(key)
    cursor.execute("DELETE FROM namespaces WHERE name = ?", (key,))
    self._db.conn.commit()

  def __iter__(self) -> typing.Iterator[str]:
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT name FROM namespaces")
    return (r[0] for r in cursor.fetchall())

  def __len__(self) -> int:
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM namespaces")
    return cursor.fetchone()[0]

  def __contains__(self, key: object) -> bool:
    if not isinstance(key, str):
      return False
    if "+" in key or "-" in key:
      tokens = re.findall(r"[A-Za-z0-9_]+", key)
      for token in tokens:
        if token not in self:
          return False
      return True
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT 1 FROM namespaces WHERE name = ?", (key,))
    return cursor.fetchone() is not None


class SQLiteClassListsDict(collections.abc.MutableMapping):
  """Dictionary-like proxy for class lists in SQLite."""

  def __init__(self, db: "TaxonomyDatabase"):
    self._db = db

  def __ior__(self, other: typing.Any) -> typing.Any:
    self.update(other)
    return self

  def __or__(self, other: typing.Any) -> typing.Any:
    res = dict(self)
    res.update(other)
    return res

  def __getitem__(self, key: str) -> namespace.ClassList:
    cursor = self._db.conn.cursor()
    cursor.execute(
        "SELECT id, namespace_name, classes_uids "
        "FROM class_lists WHERE name = ?",
        (key,),
    )
    row = cursor.fetchone()
    if row is None:
      raise KeyError(key)
    _, ns_name, blob = row
    if not blob:
      return namespace.ClassList(namespace=ns_name, classes=())
    uids = deserialize_uids(blob)
    if len(uids) < 1000:
      placeholders = ",".join("?" for _ in uids)
      cursor.execute(
          f"SELECT uid, value FROM strings WHERE uid IN ({placeholders})",
          [int(x) for x in uids],
      )
      uid_to_str = {uid: val for uid, val in cursor.fetchall()}
      classes = tuple(uid_to_str[uid] for uid in uids if uid in uid_to_str)
    else:
      uid_to_str = self._db.get_uid_to_str()
      classes = tuple(uid_to_str[uid] for uid in uids if uid in uid_to_str)
    return namespace.ClassList(namespace=ns_name, classes=classes)

  def __setitem__(self, key: str, value: namespace.ClassList) -> None:
    if not isinstance(value, namespace.ClassList):
      raise TypeError("Value must be a ClassList object")
    cursor = self._db.conn.cursor()
    classes = list(value.classes)
    uids_dict = get_or_insert_strings(self._db.conn, classes)
    uids = np.array([uids_dict[cls] for cls in classes], dtype=np.int32)
    classes_uids_blob = serialize_uids(uids)

    cursor.execute("DELETE FROM class_lists WHERE name = ?", (key,))
    cursor.execute(
        "INSERT INTO class_lists (name, namespace_name, classes_uids) VALUES"
        " (?, ?, ?)",
        (key, value.namespace, classes_uids_blob),
    )
    self._db.clear_string_caches()
    self._db.conn.commit()

  def __delitem__(self, key: str) -> None:
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT 1 FROM class_lists WHERE name = ?", (key,))
    if cursor.fetchone() is None:
      raise KeyError(key)
    cursor.execute("DELETE FROM class_lists WHERE name = ?", (key,))
    self._db.conn.commit()

  def __iter__(self) -> typing.Iterator[str]:
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT name FROM class_lists")
    return (r[0] for r in cursor.fetchall())

  def __len__(self) -> int:
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM class_lists")
    return cursor.fetchone()[0]

  def __contains__(self, key: object) -> bool:
    if not isinstance(key, str):
      return False
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT 1 FROM class_lists WHERE name = ?", (key,))
    return cursor.fetchone() is not None


class SQLiteMappingsDict(collections.abc.MutableMapping):
  """Dictionary-like proxy for mappings in SQLite."""

  def __init__(self, db: "TaxonomyDatabase"):
    self._db = db

  def __ior__(self, other: typing.Any) -> typing.Any:
    self.update(other)
    return self

  def __or__(self, other: typing.Any) -> typing.Any:
    res = dict(self)
    res.update(other)
    return res

  def __getitem__(self, key: str) -> namespace.Mapping:
    cursor = self._db.conn.cursor()
    cursor.execute(
        "SELECT id, source_namespace_name, target_namespace_name,"
        " is_default, source_uids, target_uids FROM mappings WHERE name = ?",
        (key,),
    )
    row = cursor.fetchone()
    if row is None:
      raise KeyError(key)
    _, src_ns_name, tgt_ns_name, is_default, src_blob, tgt_blob = row

    if not src_blob or not tgt_blob:
      explicit_pairs = {}
    else:
      src_uids = deserialize_uids(src_blob)
      tgt_uids = deserialize_uids(tgt_blob)
      all_uids = list(
          set(int(x) for x in src_uids) | set(int(x) for x in tgt_uids)
      )

      if len(all_uids) < 1000:
        placeholders = ",".join("?" for _ in all_uids)
        cursor.execute(
            f"SELECT uid, value FROM strings WHERE uid IN ({placeholders})",
            all_uids,
        )
        uid_to_str = {uid: val for uid, val in cursor.fetchall()}
      else:
        uid_to_str = self._db.get_uid_to_str()

      explicit_pairs = {
          uid_to_str[src]: uid_to_str[tgt]
          for src, tgt in zip(src_uids, tgt_uids)
          if src in uid_to_str and tgt in uid_to_str
      }

    try:
      src_ns = self._db.namespaces[src_ns_name]
      tgt_ns = self._db.namespaces[tgt_ns_name]
      common = src_ns.classes.intersection(tgt_ns.classes)
      mapped_pairs = {x: x for x in common}
    except KeyError:
      mapped_pairs = {}

    mapped_pairs.update(explicit_pairs)

    return namespace.Mapping(
        source_namespace=src_ns_name,
        target_namespace=tgt_ns_name,
        mapped_pairs=mapped_pairs,
        default=bool(is_default),
    )

  def __setitem__(self, key: str, value: namespace.Mapping) -> None:
    if not isinstance(value, namespace.Mapping):
      raise TypeError("Value must be a Mapping object")
    cursor = self._db.conn.cursor()

    try:
      src_ns = self._db.namespaces[value.source_namespace]
      tgt_ns = self._db.namespaces[value.target_namespace]
      common = src_ns.classes.intersection(tgt_ns.classes)
    except KeyError:
      common = set()

    pairs_to_save = {}
    for k, v in value.mapped_pairs.items():
      if k == v and k in common:
        continue
      pairs_to_save[k] = v

    all_strings = list(pairs_to_save.keys()) + list(pairs_to_save.values())
    uids_dict = get_or_insert_strings(self._db.conn, all_strings)

    if pairs_to_save:
      src_uids = [uids_dict[src] for src in pairs_to_save.keys()]
      tgt_uids = [uids_dict[tgt] for tgt in pairs_to_save.values()]
      source_uids_blob = serialize_uids(src_uids)
      target_uids_blob = serialize_uids(tgt_uids)
    else:
      source_uids_blob = None
      target_uids_blob = None

    cursor.execute("DELETE FROM mappings WHERE name = ?", (key,))
    cursor.execute(
        "INSERT INTO mappings (name, source_namespace_name,"
        " target_namespace_name, is_default, source_uids, target_uids)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            key,
            value.source_namespace,
            value.target_namespace,
            int(value.default),
            source_uids_blob,
            target_uids_blob,
        ),
    )
    self._db.clear_string_caches()
    self._db.conn.commit()

  def __delitem__(self, key: str) -> None:
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT 1 FROM mappings WHERE name = ?", (key,))
    if cursor.fetchone() is None:
      raise KeyError(key)
    cursor.execute("DELETE FROM mappings WHERE name = ?", (key,))
    self._db.conn.commit()

  def __iter__(self) -> typing.Iterator[str]:
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT name FROM mappings")
    return (r[0] for r in cursor.fetchall())

  def __len__(self) -> int:
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM mappings")
    return cursor.fetchone()[0]

  def __contains__(self, key: object) -> bool:
    if not isinstance(key, str):
      return False
    cursor = self._db.conn.cursor()
    cursor.execute("SELECT 1 FROM mappings WHERE name = ?", (key,))
    return cursor.fetchone() is not None


class TaxonomyDatabase:
  """The taxonomy database structure based on SQLite backend."""

  def __init__(
      self,
      conn: sqlite3.Connection | dict[str, namespace.Namespace] | None = None,
      class_lists: dict[str, namespace.ClassList] | None = None,
      mappings: dict[str, namespace.Mapping] | None = None,
  ):
    self._uid_to_str_cache = None
    if isinstance(conn, sqlite3.Connection):
      self.conn = conn
    else:
      # If namespaces dictionary (or None) is passed as first argument,
      # initialize with an in-memory SQLite connection!
      self.conn = sqlite3.connect(":memory:", check_same_thread=False)
      create_tables(self.conn)

      self.namespaces = SQLiteNamespacesDict(self)
      self.class_lists = SQLiteClassListsDict(self)
      self.mappings = SQLiteMappingsDict(self)

      if conn is not None:
        for k, v in conn.items():
          self.namespaces[k] = v
      if class_lists is not None:
        for k, v in class_lists.items():
          self.class_lists[k] = v
      if mappings is not None:
        for k, v in mappings.items():
          self.mappings[k] = v
      return

    self.namespaces = SQLiteNamespacesDict(self)
    self.class_lists = SQLiteClassListsDict(self)
    self.mappings = SQLiteMappingsDict(self)

  def get_uid_to_str(self) -> dict[int, str]:
    if self._uid_to_str_cache is None:
      cursor = self.conn.cursor()
      cursor.execute("SELECT uid, value FROM strings")
      self._uid_to_str_cache = {uid: val for uid, val in cursor.fetchall()}
    return self._uid_to_str_cache

  def clear_string_caches(self):
    self._uid_to_str_cache = None


def validate_taxonomy_database(taxonomy_database: TaxonomyDatabase) -> None:
  """Validate the taxonomy database.

  This ensures that all class lists, namespaces, and mappings are consistent.

  Args:
    taxonomy_database: A taxonomy database structure to validate.

  Raises:
    ValueError: if a class list or mapping contains a class not in the
      namespace.
    KeyError: if a namespace or class list is not found.
  """
  cursor = taxonomy_database.conn.cursor()

  # 1. Fetch unknown_uid if it exists
  cursor.execute(
      "SELECT uid FROM strings WHERE value = ?", (namespace.UNKNOWN_LABEL,)
  )
  row = cursor.fetchone()
  unknown_uid = row[0] if row else None

  # 2. Fetch all base namespaces name -> classes_uids set
  cursor.execute("SELECT name, classes_uids FROM namespaces")
  namespaces_uids = {}
  for name, blob in cursor.fetchall():
    uids = deserialize_uids(blob)
    namespaces_uids[name] = set(int(x) for x in uids)

  # 3. Validate Mappings
  cursor.execute(
      "SELECT name, source_namespace_name, target_namespace_name,"
      " source_uids, target_uids FROM mappings"
  )
  for m_name, src_ns_name, tgt_ns_name, src_blob, tgt_blob in cursor.fetchall():
    if src_ns_name not in namespaces_uids:
      raise KeyError(
          f"Source namespace {src_ns_name} for mapping {m_name} not found."
      )
    if tgt_ns_name not in namespaces_uids:
      raise KeyError(
          f"Target namespace {tgt_ns_name} for mapping {m_name} not found."
      )

    src_set = namespaces_uids[src_ns_name]
    tgt_set = namespaces_uids[tgt_ns_name]

    if src_blob and tgt_blob:
      src_uids = set(int(x) for x in deserialize_uids(src_blob))
      tgt_uids = set(int(x) for x in deserialize_uids(tgt_blob))

      missing_src = src_uids - src_set
      if missing_src:
        raise ValueError(
            f"Mapping {m_name} contains a source class not in "
            f"the namespace ({src_ns_name})."
        )

      missing_tgt = tgt_uids - tgt_set
      if missing_tgt:
        raise ValueError(
            f"Mapping {m_name} contains a target class not in "
            f"the namespace ({tgt_ns_name})."
        )

  # 4. Validate Class Lists
  cursor.execute("SELECT name, namespace_name, classes_uids FROM class_lists")
  for cl_name, ns_name, blob in cursor.fetchall():
    if "+" in ns_name or "-" in ns_name:
      # Fallback for algebraic namespaces
      ns_classes = taxonomy_database.namespaces[ns_name].classes
      cl_classes = taxonomy_database.class_lists[cl_name].classes
      if set(cl_classes) - ns_classes - {namespace.UNKNOWN_LABEL}:
        raise ValueError(
            f"ClassList {cl_name} contains a class not in "
            f"the namespace ({ns_name})."
        )
    else:
      if ns_name not in namespaces_uids:
        raise KeyError(
            f"Namespace {ns_name} for class list {cl_name} not found."
        )

      ns_set = namespaces_uids[ns_name]
      if blob:
        cl_uids = set(int(x) for x in deserialize_uids(blob))
        allowed_uids = ns_set
        if unknown_uid is not None:
          allowed_uids = allowed_uids | {unknown_uid}

        missing_cl = cl_uids - allowed_uids
        if missing_cl:
          raise ValueError(
              f"ClassList {cl_name} contains a class not in "
              f"the namespace ({ns_name})."
          )


def dump_db(taxonomy_database: TaxonomyDatabase, validate: bool = True) -> str:
  """Serialize SQLite taxonomy database back to a JSON-formatted string."""
  if validate:
    validate_taxonomy_database(taxonomy_database)

  data = {
      "namespaces": {
          name: {"classes": sorted(list(ns.classes))}
          for name, ns in sorted(taxonomy_database.namespaces.items())
      },
      "class_lists": {
          name: {"namespace": cl.namespace, "classes": list(cl.classes)}
          for name, cl in sorted(taxonomy_database.class_lists.items())
      },
      "mappings": {
          name: {
              "source_namespace": m.source_namespace,
              "target_namespace": m.target_namespace,
              "default": m.default,
              "mapped_pairs": {
                  k: v
                  for k, v in sorted(m.mapped_pairs.items())
                  if not (
                      k == v
                      and k
                      in (
                          (  # pylint: disable=g-long-ternary
                              taxonomy_database.namespaces[
                                  m.source_namespace
                              ].classes
                              & taxonomy_database.namespaces[
                                  m.target_namespace
                              ].classes
                          )
                          if m.source_namespace in taxonomy_database.namespaces
                          and m.target_namespace in taxonomy_database.namespaces
                          else set()
                      )
                  )
              },
          }
          for name, m in sorted(taxonomy_database.mappings.items())
      },
  }
  return json.dumps(
      data,
      indent=2,
      sort_keys=True,
  )


_in_memory_dbs: dict[tuple[str, bool], TaxonomyDatabase] = {}


def _resolve_db_path(path_str: str) -> str:
  """Resolve the database path to an absolute path."""
  try:
    abs_path = path_utils.get_absolute_path(path_str)
  except (ValueError, OSError):
    abs_path = None

  file_exists = os.path.exists(path_str)
  resolved_path = path_str
  if not file_exists and abs_path is not None:
    try:
      if os.path.exists(abs_path):
        file_exists = True
        resolved_path = str(abs_path)
    except OSError:
      pass

  if not file_exists:
    raise FileNotFoundError(f"Database file '{resolved_path}' not found.")
  return resolved_path


@functools.cache
def _load_db_cached(
    path_str: str,
    validate: bool,
    read_only: bool,
) -> TaxonomyDatabase:
  """Load the taxonomy database from an SQLite file, with caching."""
  resolved_path = _resolve_db_path(path_str)
  if read_only:
    # Open in read-only mode using URI
    abs_resolved_path = os.path.abspath(resolved_path)
    conn = sqlite3.connect(
        f"file:{abs_resolved_path}?mode=ro", uri=True, check_same_thread=False
    )
  else:
    conn = sqlite3.connect(resolved_path, check_same_thread=False)
    create_tables(conn)
  taxonomy_database = TaxonomyDatabase(conn)

  if validate:
    validate_taxonomy_database(taxonomy_database)
  return taxonomy_database


def _load_in_memory(path_str: str, validate: bool) -> TaxonomyDatabase:
  """Load the taxonomy database into an in-memory copy."""
  resolved_path = _resolve_db_path(path_str)
  abs_resolved_path = os.path.abspath(resolved_path)
  source_conn = sqlite3.connect(
      f"file:{abs_resolved_path}?mode=ro", uri=True, check_same_thread=False
  )
  conn = sqlite3.connect(":memory:", check_same_thread=False)
  source_conn.backup(conn)
  source_conn.close()

  taxonomy_database = TaxonomyDatabase(conn)
  if validate:
    validate_taxonomy_database(taxonomy_database)
  return taxonomy_database


def load_db(
    path: os.PathLike[str] | str = TAXONOMY_DATABASE_FILENAME,
    validate: bool = True,
    read_only: bool = True,
    in_memory: bool = False,
) -> TaxonomyDatabase:
  """Load the taxonomy database.

  This loads the taxonomy database from the given SQLite file.

  Args:
    path: The file to load.
    validate: If true, it validates the database.
    read_only: If true and loading an SQLite file, opens it in read-only mode.
    in_memory: If true, loads the database into an in-memory copy.

  Returns:
    The taxonomy database.
  Raises:
    FileNotFoundError: if the database file does not exist.
  """
  path_str = str(path)
  key_in_mem = (path_str, validate)
  if in_memory or key_in_mem in _in_memory_dbs:
    if key_in_mem not in _in_memory_dbs:
      _in_memory_dbs[key_in_mem] = _load_in_memory(path_str, validate)
    return _in_memory_dbs[key_in_mem]
  return _load_db_cached(path_str, validate, read_only)
