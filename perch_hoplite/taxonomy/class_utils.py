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

"""Convenience utilities for handling class lists and species labels."""

from collections.abc import Sequence
import logging
from typing import Literal

from perch_hoplite.taxonomy import namespace_db


def get_class_lists(species_class_list_name: str, add_taxonomic_labels: bool):
  """Get the number of classes for the target class outputs."""
  db = namespace_db.load_db()
  species_classes = db.class_lists[species_class_list_name]
  class_lists = {
      "label": species_classes,
  }
  if add_taxonomic_labels:
    for name in ["genus", "family", "order"]:
      mapping_name = f"{species_classes.namespace}_to_{name}"
      mapping = db.mappings[mapping_name]
      taxa_class_list = species_classes.apply_namespace_mapping(mapping)
      class_lists[name] = taxa_class_list
  return class_lists



def convert_species_to_namespace(
    species: str | Sequence[str],
    target_namespace: str = 'inat2024',
    *,
    source_namespace: str | None = None,
    taxonomy_database: namespace_db.TaxonomyDatabase | None = None,
    errors: Literal['raise', 'warn', 'ignore'] = 'raise',
) -> str | None | list[str | None]:
  """Convert unknown-provenance species labels to a target taxonomy.

  Search registered namespaces for each exact label and use stored mappings
  to the requested target namespace. All candidate source namespaces must
  agree on the output: conflicting conversions are ambiguous and are never
  resolved by an arbitrary namespace search order.

  Args:
    species: One label or a sequence of labels; sequence results retain input
      ordering and duplicates. Unresolvable entries become None when
      errors='warn' or 'ignore'.
    target_namespace: Destination taxonomy (defaults to iNat2024).
    source_namespace: Optional known source taxonomy to disambiguate labels.
    taxonomy_database: Optional database, otherwise the cached default.
    errors: Whether to raise ValueError, log a warning, or silently return
      None when no mapping exists or candidate mappings conflict.

  Returns:
    A single label/None for a single input; a list of labels/Nones for a
    sequence. No approximate or fuzzy species matching is performed.
  """
  if errors not in ('raise', 'warn', 'ignore'):
    raise ValueError("errors must be 'raise', 'warn', or 'ignore'")

  single = isinstance(species, str)
  if single:
    labels = [species]
  elif isinstance(species, Sequence):
    labels = list(species)
  else:
    raise TypeError('species must be a string or sequence of strings')
  if not all(isinstance(label, str) for label in labels):
    raise TypeError('all species labels must be strings')
  if not labels:
    return []

  db = taxonomy_database if taxonomy_database is not None else namespace_db.load_db()
  if target_namespace not in db.namespaces:
    raise ValueError(f'Unknown target namespace: {target_namespace}')
  if source_namespace is not None and source_namespace not in db.namespaces:
    raise ValueError(f'Unknown source namespace: {source_namespace}')

  unique_labels = tuple(dict.fromkeys(labels))
  source_matches = {label: set() for label in unique_labels}
  candidate_targets = {label: set() for label in unique_labels}
  target_classes = db.namespaces[target_namespace].classes

  namespaces_to_check = (
      (source_namespace,) if source_namespace is not None else tuple(db.namespaces)
  )
  for ns_name in namespaces_to_check:
    ns_classes = db.namespaces[ns_name].classes
    for label in unique_labels:
      if label in ns_classes:
        source_matches[label].add(ns_name)
        if ns_name == target_namespace:
          candidate_targets[label].add(label)

  # Only load mappings whose source contains a requested label, rather than
  # materializing every mapping in the (potentially large) taxonomy database.
  matching_namespaces = set().union(*source_matches.values())
  cursor = db.conn.cursor()
  cursor.execute(
      'SELECT name, source_namespace_name FROM mappings '
      'WHERE target_namespace_name = ?',
      (target_namespace,),
  )
  for mapping_name, mapped_source in cursor.fetchall():
    if mapped_source not in matching_namespaces:
      continue
    mapped_pairs = db.mappings[mapping_name].mapped_pairs
    for label in unique_labels:
      if mapped_source in source_matches[label] and label in mapped_pairs:
        mapped_label = mapped_pairs[label]
        if mapped_label in target_classes:
          candidate_targets[label].add(mapped_label)

  conversions = {}
  for label in unique_labels:
    targets = candidate_targets[label]
    if len(targets) == 1:
      conversions[label] = next(iter(targets))
      continue

    if targets:
      problem = (
          f'Ambiguous species {label!r} for {target_namespace}: '
          f'{sorted(targets)}'
      )
    else:
      problem = f'No mapping for species {label!r} to {target_namespace}'
    if errors == 'raise':
      raise ValueError(problem)
    if errors == 'warn':
      logging.warning('%s', problem)
    conversions[label] = None

  result = [conversions[label] for label in labels]
  return result[0] if single else result
