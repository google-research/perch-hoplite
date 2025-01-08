# coding=utf-8
# Copyright 2024 The Perch Authors.
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

"""Base class for a searchable embeddings database."""

import abc
import dataclasses
import enum
from typing import Iterator, Sequence

from ml_collections import config_dict
import numpy as np


class LabelType(enum.Enum):
  NEGATIVE = 0
  POSITIVE = 1


@dataclasses.dataclass
class Label:
  """Label for an embedding.

  Attributes:
    embedding_id: Unique integer ID for the embedding this label applies to.
    label: Label string.
    type: Type of label (positive, negative, etc).
    provenance: Freeform field describing the annotation (eg, labeler name,
      model identifier for pseudolabels, etc).
  """

  embedding_id: int
  label: str
  type: LabelType
  provenance: str


@dataclasses.dataclass
class EmbeddingSource:
  """Source information for an embedding."""

  dataset_name: str
  source_id: str
  offsets: np.ndarray

  def __eq__(self, other):
    return (
        self.dataset_name == other.dataset_name
        and self.source_id == other.source_id
        and np.array_equal(self.offsets, other.offsets)
    )


@dataclasses.dataclass
class EmbeddingMetadata:
  """Convenience class for converting dataclasses to/from ConfigDict."""

  def to_config_dict(self) -> config_dict.ConfigDict:
    """Convert to a config dict."""
    return config_dict.ConfigDict(dataclasses.asdict(self))

  @classmethod
  def from_config_dict(
      cls, config: config_dict.ConfigDict
  ) -> 'EmbeddingMetadata':
    """Convert from a config dict."""
    return cls(**config)


class HopliteDBInterface(abc.ABC):
  """Interface for searchable embeddings database with metadata.

  The database consists of a table of embeddings with a unique id for each
  embedding, and some metadata for linking the embedding to its source.
  Additionally, a Key-Value table of ConfigDict objects is used to store
  arbitrary metadata associated with the database.

  Methods are split into 'Base' methods and 'Composite' methods. Base methods
  must be implemented for any implementation. Composite methods have a default
  implementation using the base methods, but may benefit from implementation-
  specific optimizations.
  """

  # Base methods

  @classmethod
  @abc.abstractmethod
  def create(cls, **kwargs):
    """Connect to and, if needed, initialize the database."""

  @abc.abstractmethod
  def commit(self) -> None:
    """Commit any pending transactions to the database."""

  @abc.abstractmethod
  def thread_split(self) -> 'HopliteDBInterface':
    """Get a new instance of the database with the same contents.

    For example, SQLite DB's need a distinct object in each thread.
    """

  @abc.abstractmethod
  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    """Insert a key-value pair into the metadata table."""

  @abc.abstractmethod
  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    """Get a key-value pair from the metadata table.

    Args:
      key: String for metadata key to retrieve. If None, returns all metadata.

    Returns:
      ConfigDict containing the metadata.
    """

  @abc.abstractmethod
  def get_dataset_names(self) -> Sequence[str]:
    """Get all dataset names in the database."""

  @abc.abstractmethod
  def get_embedding_ids(self) -> np.ndarray:
    # TODO(tomdenton): Make this return an iterator, with optional shuffling.
    """Get all embedding IDs in the database."""

  @abc.abstractmethod
  def insert_embedding(
      self, embedding: np.ndarray, source: EmbeddingSource
  ) -> int:
    """Add an embedding to the database."""

  @abc.abstractmethod
  def get_embedding(self, embedding_id: int) -> np.ndarray:
    """Retrieve an embedding from the database."""

  @abc.abstractmethod
  def get_embedding_source(self, embedding_id: int) -> EmbeddingSource:
    """Get the source corresponding to the given embedding_id."""

  @abc.abstractmethod
  def get_embeddings_by_source(
      self,
      dataset_name: str,
      source_id: str | None,
      offsets: np.ndarray | None = None,
  ) -> np.ndarray:
    """Get the embedding IDs for all embeddings matching the source.

    Args:
      dataset_name: The name of the dataset to search.
      source_id: The ID of the source to search. If None, all sources are
        searched.
      offsets: The offsets of the source to search. If None, all offsets are
        searched.

    Returns:
      A list of embedding IDs matching the indicated source parameters.
    """

  @abc.abstractmethod
  def insert_label(self, label: Label, skip_duplicates: bool = False) -> bool:
    """Add a label to the db.

    Args:
      label: The label to insert.
      skip_duplicates: If True, and the label already exists, return False.
        Otherwise, the label is inserted regardless of duplicates.

    Returns:
      True if the label was inserted, False if it was a duplicate and
      skip_duplicates was True.
    Raises:
      ValueError if the label type or provenance is not set.
    """

  @abc.abstractmethod
  def embedding_dimension(self) -> int:
    """Get the embedding dimension."""

  @abc.abstractmethod
  def get_embeddings_by_label(
      self,
      label: str,
      label_type: LabelType | None = LabelType.POSITIVE,
      provenance: str | None = None,
  ) -> np.ndarray:
    """Find embeddings by label.

    Args:
      label: Label string to search for.
      label_type: Type of label to return. If None, returns all labels
        regardless of Type.
      provenance: If provided, filters to the target provenance value.

    Returns:
      An array of unique embedding id's matching the label.
    """
    # TODO(tomdenton): Allow fetching by dataset_name.

  @abc.abstractmethod
  def get_labels(self, embedding_id: int) -> Sequence[Label]:
    """Get all labels for the indicated embedding_id."""

  @abc.abstractmethod
  def get_classes(self) -> Sequence[str]:
    """Get all distinct classes (label strings) in the database."""

  @abc.abstractmethod
  def get_class_counts(
      self, label_type: LabelType = LabelType.POSITIVE
  ) -> dict[str, int]:
    """Count the number of occurences of each class in the database.

    Classes with zero matching occurences are still included in the result.

    Args:
      label_type: Type of label to count. By default, counts positive labels.
    """

  # Composite methods

  def get_one_embedding_id(self) -> int:
    """Get an arbitrary embedding id from the database."""
    return self.get_embedding_ids()[0]

  def count_embeddings(self) -> int:
    """Return a count of all embeddings in the database."""
    return len(self.get_embedding_ids())

  def count_classes(self) -> int:
    """Return a count of all distinct classes in the database."""
    return len(self.get_classes())

  def get_embeddings(
      self, embedding_ids: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Get an array of embeddings for the indicated IDs.

    Note that the embeddings may not be returned in the same order as the
    provided embedding_id's. Thus, we suggest the usage:
    ```
    idxes, embeddings = db.get_embeddings(idxes)
    ```

    Args:
      embedding_ids: 1D array of embedding id's.

    Returns:
      Permuted array of embedding_id's and embeddings.
    """
    embeddings = [self.get_embedding(int(idx)) for idx in embedding_ids]
    return embedding_ids, np.array(embeddings)

  def get_embedding_sources(
      self, embedding_ids: np.ndarray
  ) -> tuple[EmbeddingSource, ...]:
    """Get an array of embedding sources for the indicated IDs."""
    return tuple(self.get_embedding_source(int(idx)) for idx in embedding_ids)

  def random_batched_iterator(
      self,
      ids: np.ndarray,
      batch_size: int,
      rng: np.random.RandomState,
  ) -> Iterator[np.ndarray]:
    """Yields batches embedding ids, shuffled after each of unlimited epochs."""
    if batch_size > len(ids):
      raise ValueError('Not enough ids to fill a batch.')
    rng.shuffle(ids)
    q = 0
    while True:
      if q + batch_size > len(ids):
        partial_batch = ids[q : len(ids)].copy()
        overflow = batch_size - len(partial_batch)
        rng.shuffle(ids)
        overflow_batch = ids[:overflow]
        batch = np.concatenate([partial_batch, overflow_batch], axis=0)
        q = overflow
      else:
        batch = ids[q : q + batch_size]
        q += batch_size
      yield batch.copy()
