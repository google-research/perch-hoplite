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

"""Database of bioacoustic label domains."""

import dataclasses
import functools
import json
import os
import typing

from perch_hoplite import path_utils
from perch_hoplite.taxonomy import namespace

TAXONOMY_DATABASE_FILENAME = "taxonomy/taxonomy_database.json"


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


@dataclasses.dataclass
class TaxonomyDatabase:
  namespaces: dict[str, namespace.Namespace]
  class_lists: dict[str, namespace.ClassList]
  mappings: dict[str, namespace.Mapping]


def validate_taxonomy_database(taxonomy_database: TaxonomyDatabase) -> None:
  """Validate the taxonomy database.

  This ensures that all class lists, namespaces, and mappings are consistent.

  Args:
    taxonomy_database: A taxonomy database structure to validate.

  Raises:
    ValueError or KeyError when the database is invalid.
  """
  namespaces = taxonomy_database.namespaces

  for mapping_name, mapping in taxonomy_database.mappings.items():
    if (
        set(mapping.mapped_pairs.keys())
        - namespaces[mapping.source_namespace].classes
    ):
      raise ValueError(
          f"Mapping {mapping_name} contains a source class not in "
          f"the namespace ({mapping.source_namespace})."
      )
    if (
        set(mapping.mapped_pairs.values())
        - namespaces[mapping.target_namespace].classes
    ):
      raise ValueError(
          f"Mapping {mapping_name} contains a target class not in "
          f"the namespace ({mapping.target_namespace})."
      )

  for class_name, class_list in taxonomy_database.class_lists.items():
    classes = class_list.classes
    if (
        set(classes)
        - namespaces[class_list.namespace].classes
        - {namespace.UNKNOWN_LABEL}
    ):
      raise ValueError(
          f"ClassList {class_name} contains a class not in "
          f"the namespace ({class_list.namespace})."
      )


def load_taxonomy_database(
    taxonomy_database: dict[str, typing.Any],
) -> TaxonomyDatabase:
  """Construct a taxonomy database from a dictionary.

  Args:
    taxonomy_database: The database as loaded from a JSON file.

  Returns:
    A taxonomy database.

  Raises:
    TypeError when the database contains unknown keys.
  """
  namespaces = {
      name: namespace.Namespace(
          classes=frozenset(namespace_.pop("classes")), **namespace_
      )
      for name, namespace_ in taxonomy_database.pop("namespaces").items()
  }
  class_lists = {
      name: namespace.ClassList(
          classes=tuple(class_list.pop("classes")), **class_list
      )
      for name, class_list in taxonomy_database.pop("class_lists").items()
  }
  mappings = {
      name: namespace.Mapping(**mapping)
      for name, mapping in taxonomy_database.pop("mappings").items()
  }
  return TaxonomyDatabase(
      namespaces=namespaces,
      class_lists=class_lists,
      mappings=mappings,
      **taxonomy_database,
  )


class TaxonomyDatabaseEncoder(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, frozenset):
      return sorted(o)
    return super().default(o)


def dump_db(taxonomy_database: TaxonomyDatabase, validate: bool = True) -> str:
  if validate:
    validate_taxonomy_database(taxonomy_database)
  return json.dumps(
      dataclasses.asdict(taxonomy_database),
      cls=TaxonomyDatabaseEncoder,
      indent=2,
      sort_keys=True,
  )


@functools.cache
def load_db(
    path: os.PathLike[str] | str = TAXONOMY_DATABASE_FILENAME,
    validate: bool = True,
) -> TaxonomyDatabase:
  """Load the taxonomy database.

  This loads the taxonomy database from the given JSON file. It converts the
  database into Python data structures and optionally validates that the
  database is consistent.

  Args:
    path: The JSON file to load.
    validate: If true, it validates the database.

  Returns:
    The taxonomy database.
  """
  with path_utils.open_file(path, "r") as f:
    data = json.load(f)
  taxonomy_database = load_taxonomy_database(data)
  if validate:
    validate_taxonomy_database(taxonomy_database)
  return taxonomy_database
