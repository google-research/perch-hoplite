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

"""Optional, lazy USearch vector index wrapper for Hoplite DB backends."""

from collections.abc import Sequence
import itertools
import types
from typing import Any

from absl import logging
from etils import epath
from ml_collections import config_dict
import numpy as np
from usearch import index as uindex

UINDEX_FILENAME = 'usearch.index'
USEARCH_CONFIG_KEY = 'usearch_config'
DB_OPTIONS_KEY = 'db_options'
USEARCH_DTYPES = types.MappingProxyType({
    'float16': uindex.ScalarKind.F16,
})


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


def resolve_usearch_config(
    metadata: config_dict.ConfigDict,
    usearch_cfg: config_dict.ConfigDict | None,
    *,
    use_embeddings: bool = True,
    readonly: bool = False,
) -> tuple[config_dict.ConfigDict | None, bool, bool]:
  """Resolve USearch config against existing database metadata.

  Args:
    metadata: All key-value entries currently stored in `hoplite_metadata`.
    usearch_cfg: Caller-supplied USearch configuration, or None.
    use_embeddings: Whether embeddings are enabled. Passing `False` opts into
      creating or opening a database without a USearch index.
    readonly: Whether the database is being opened in read-only mode.

  Returns:
    A tuple `(resolved_cfg, should_write_usearch_cfg, should_write_no_emb)`:
      - `resolved_cfg`: The active USearch config, or None if embeddings are
        disabled.
      - `should_write_usearch_cfg`: True if `resolved_cfg` should be persisted
        to `hoplite_metadata` under `USEARCH_CONFIG_KEY` (and `db_options`
        updated if previously disabled).
      - `should_write_no_emb`: True if `db_options={'use_embeddings': False}`
        should be persisted to `hoplite_metadata`.

  Raises:
    ValueError: If the requested configuration conflicts with existing metadata,
      if a new embeddings-enabled database is opened without `usearch_cfg`, or
      if attempting to enable embeddings on a read-only database.
  """
  has_existing_cfg = USEARCH_CONFIG_KEY in metadata
  db_options = metadata.get(DB_OPTIONS_KEY)
  existing_no_embeddings = db_options is not None and not db_options.get(
      'use_embeddings', True
  )

  if not use_embeddings:
    if usearch_cfg is not None:
      raise ValueError(
          'Cannot specify both `usearch_cfg` and `use_embeddings=False`.'
      )
    if has_existing_cfg:
      logging.warning(
          'Opening an embeddings-enabled database with `use_embeddings=False`;'
          ' embedding lookups, search, and index updates will be disabled for'
          ' this connection.'
      )
      return None, False, False
    return None, False, not existing_no_embeddings

  if has_existing_cfg:
    existing_cfg = metadata[USEARCH_CONFIG_KEY]
    if usearch_cfg is not None and existing_cfg != usearch_cfg:
      raise ValueError(
          'A usearch_cfg was provided, but a different one already exists in'
          ' the DB.'
      )
    return existing_cfg, False, False

  if existing_no_embeddings:
    if usearch_cfg is not None:
      if readonly:
        raise ValueError(
            'Cannot enable embeddings on a read-only database that was created'
            ' without embeddings.'
        )
      return usearch_cfg, True, False
    return None, False, False

  if usearch_cfg is None:
    raise ValueError(
        'No usearch_cfg was found in DB and none was provided. Pass'
        ' `usearch_cfg` to create an embeddings DB, or `use_embeddings=False`'
        ' to create a database without embeddings.'
    )

  return usearch_cfg, True, False


class USearchVectorStore:
  """Manages an optional, lazily-created USearch index on disk.

  When `usearch_cfg` is provided and `index_path` does not yet exist, no index
  file is written to disk until at least one embedding is added and `save()` is
  called. When `usearch_cfg` is None, embedding storage is disabled.

  Attributes:
    index_path: Path to the USearch index file on disk.
    usearch_cfg: USearch configuration, or None if embeddings are disabled.
    readonly: Whether mutations are disallowed.
  """

  def __init__(
      self,
      index_path: epath.Path,
      usearch_cfg: config_dict.ConfigDict | None,
      readonly: bool = False,
  ) -> None:
    """Initializes the vector store."""
    self.index_path = index_path
    self.usearch_cfg = usearch_cfg
    self.readonly = readonly
    self._ui: uindex.Index | None = None
    self._loaded: bool = False
    self._updated: bool = False

    if usearch_cfg is not None:
      if usearch_cfg.dtype not in USEARCH_DTYPES:
        raise ValueError(
            f'Unsupported USearch dtype: {usearch_cfg.dtype!r}. Supported:'
            f' {sorted(USEARCH_DTYPES)}.'
        )
      self._embedding_dim: int | None = int(usearch_cfg.embedding_dim)
      self._embedding_dtype: type[Any] | None = np.dtype(usearch_cfg.dtype).type
      if index_path.exists():
        self._ui = uindex.Index(
            ndim=self._embedding_dim,
            path=index_path,
            view=True,
        )
        self._loaded = False
    else:
      self._embedding_dim = None
      self._embedding_dtype = None

  @property
  def enabled(self) -> bool:
    """Whether embedding storage is enabled for this database."""
    return self.usearch_cfg is not None

  @property
  def embedding_dim(self) -> int:
    """The configured embedding dimension."""
    if self._embedding_dim is None:
      raise ValueError('Embeddings are not enabled for this database.')
    return self._embedding_dim

  @property
  def embedding_dtype(self) -> type[Any]:
    """The configured NumPy embedding scalar dtype."""
    if self._embedding_dtype is None:
      raise ValueError('Embeddings are not enabled for this database.')
    return self._embedding_dtype

  @property
  def has_index(self) -> bool:
    """Whether an index exists in memory or on disk."""
    return self._ui is not None or self.index_path.exists()

  @property
  def ui(self) -> uindex.Index:
    """The underlying USearch Index, lazily constructed if configured."""
    if self.usearch_cfg is None:
      raise ValueError('Embeddings are not enabled for this database.')
    if self._ui is None:
      if self.index_path.exists():
        self._ui = uindex.Index(
            ndim=self.embedding_dim,
            path=self.index_path,
            view=True,
        )
        self._loaded = False
      else:
        self._ui = self._create_empty_index()
        self._loaded = True
    return self._ui

  def _create_empty_index(self) -> uindex.Index:
    """Creates a fresh in-memory USearch index without writing to disk."""
    cfg = self.usearch_cfg
    if cfg is None:
      raise ValueError('Embeddings are not enabled for this database.')
    return uindex.Index(
        ndim=int(cfg.embedding_dim),
        metric=getattr(uindex.MetricKind, cfg.metric_name),
        expansion_add=cfg.expansion_add,
        expansion_search=cfg.expansion_search,
        dtype=USEARCH_DTYPES[cfg.dtype],
        path=self.index_path,
        view=False,
    )

  def _ensure_loaded_for_write(self) -> uindex.Index:
    """Returns the underlying index loaded in memory for mutation."""
    if self.readonly:
      raise RuntimeError('Cannot modify embeddings on a read-only database.')
    ui = self.ui
    if not self._loaded:
      ui.load()
      self._loaded = True
    return ui

  def validate_embedding(self, embedding: np.ndarray) -> None:
    """Validates that embeddings are enabled and match the configured dim."""
    if not self.enabled:
      raise ValueError(
          'Cannot insert embeddings because this database was created without'
          ' an embedding index (`use_embeddings=False`).'
      )
    if embedding.shape[-1] != self.embedding_dim:
      raise ValueError(
          f'Incorrect embedding dimension. Expected {self.embedding_dim}, but'
          f' got {embedding.shape[-1]}.'
      )

  def add(self, keys: int | np.ndarray, vectors: np.ndarray) -> None:
    """Adds one or more embeddings to the USearch index."""
    self.validate_embedding(vectors)
    ui = self._ensure_loaded_for_write()
    ui.add(keys, vectors.astype(self.embedding_dtype))
    self._updated = True

  def remove(self, keys: int | Sequence[int]) -> None:
    """Removes one or more keys from the USearch index if present."""
    if not self.enabled or not self.has_index:
      return
    ui = self._ensure_loaded_for_write()
    ui.remove(keys)
    self._updated = True

  def get(self, window_id: int) -> np.ndarray:
    """Gets a single embedding vector by window ID."""
    if not self.enabled or not self.has_index:
      raise KeyError(f'Embedding vector not found for window id: {window_id}')
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

  def get_batch(self, window_ids: Sequence[int]) -> np.ndarray:
    """Gets a batch of embedding vectors by window IDs."""
    if len(window_ids) == 0:  # pylint: disable=g-explicit-length-test
      return np.empty((0, self.embedding_dim), dtype=self.embedding_dtype)
    if not self.enabled or not self.has_index:
      raise KeyError(
          f'Embedding vectors not found for window ids: {list(window_ids)}'
      )
    found = self.ui.contains(window_ids)
    if not isinstance(found, np.ndarray):
      raise RuntimeError(
          'Expected np.ndarray result from the USearch `contains()` method, but'
          f' got {type(found)}: {found}.'
      )
    if not np.all(found):
      missing = list(itertools.compress(window_ids, ~found))
      raise KeyError(f'Embedding vectors not found for window ids: {missing}')
    embeddings_batch = self.ui.get(window_ids)
    if isinstance(embeddings_batch, (tuple, list)):
      embeddings_batch = np.stack(embeddings_batch)  # pyrefly: ignore[no-matching-overload]
    if not isinstance(embeddings_batch, np.ndarray):
      raise RuntimeError(
          'Expected np.ndarray result from the USearch `get()` method, but got'
          f' {type(embeddings_batch)}: {embeddings_batch}.'
      )
    return embeddings_batch

  def count(self) -> int:
    """Returns the number of embeddings stored in the index."""
    if not self.enabled or not self.has_index:
      return 0
    return int(self.ui.size)

  def search(
      self,
      query_embedding: np.ndarray,
      count: int,
      exact: bool,
  ) -> Any:
    """Runs USearch vector search against the index."""
    if not self.enabled:
      raise ValueError('Embeddings are not enabled for this database.')
    return self.ui.search(query_embedding, count=count, exact=exact)

  def save(self) -> None:
    """Persists pending index modifications to disk."""
    if self._updated and self._ui is not None:
      self._ui.save()
      self._updated = False
