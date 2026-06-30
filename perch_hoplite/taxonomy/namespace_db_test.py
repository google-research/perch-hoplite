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

"""Tests for namespace_db."""

import io
import shutil
import sqlite3
import tempfile

from absl import logging
from etils import epath
from perch_hoplite import path_utils
from perch_hoplite.taxonomy import namespace
from perch_hoplite.taxonomy import namespace_db

from absl.testing import absltest
from absl.testing import parameterized


class NamespaceDbTest(parameterized.TestCase):

  def test_load_namespace_db(self):
    db = namespace_db.load_db()

    # Check a couple ClassLists of known size.
    self.assertIn('caples', db.class_lists)
    caples_list = db.class_lists['caples']
    self.assertEqual(caples_list.namespace, 'ebird2021')
    self.assertLen(caples_list.classes, 79)

    genus_mapping = db.mappings['ebird2021_to_genus']
    caples_genera = caples_list.apply_namespace_mapping(genus_mapping)
    self.assertEqual(caples_genera.namespace, 'ebird2021_genera')
    self.assertLen(caples_genera.classes, 62)

    family_mapping = db.mappings['ebird2021_to_family']
    caples_families = caples_list.apply_namespace_mapping(family_mapping)
    self.assertEqual(caples_families.namespace, 'ebird2021_families')
    self.assertLen(caples_families.classes, 30)

    order_mapping = db.mappings['ebird2021_to_order']
    caples_orders = caples_list.apply_namespace_mapping(order_mapping)
    self.assertEqual(caples_orders.namespace, 'ebird2021_orders')
    self.assertLen(caples_orders.classes, 11)

  def test_class_map_csv(self):
    cl = namespace.ClassList(
        'ebird2021', ('amecro', 'amegfi', 'amered', 'amerob')
    )
    cl_csv = cl.to_csv()
    with io.StringIO(cl_csv) as f:
      got_cl = namespace.ClassList.from_csv(f)
    self.assertEqual(got_cl.namespace, 'ebird2021')
    self.assertEqual(got_cl.classes, ('amecro', 'amegfi', 'amered', 'amerob'))

    # Check that writing with tf.io.gfile behaves as expected, as newline
    # behavior may be different than working with StringIO.
    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
      with epath.Path(f.name).open(mode='w') as gf:
        gf.write(cl_csv)
      with open(f.name, 'r') as f:
        got_cl = namespace.ClassList.from_csv(f.readlines())
    self.assertEqual(got_cl.namespace, 'ebird2021')
    self.assertEqual(got_cl.classes, ('amecro', 'amegfi', 'amered', 'amerob'))

  def test_namespace_class_list_closure(self):
    # Ensure that all classes in class lists appear in their namespace.
    db = namespace_db.load_db()

    all_missing_classes = set()
    for list_name, class_list in db.class_lists.items():
      missing_classes = set()
      namespace_ = db.namespaces[class_list.namespace]
      for cl in class_list.classes:
        if cl not in namespace_.classes:
          missing_classes.add(cl)
          all_missing_classes.add(cl)
      if missing_classes:
        logging.warning(
            'The classes %s in class list %s did not appear in namespace %s.',
            missing_classes,
            list_name,
            class_list.namespace,
        )
      missing_classes.discard('unknown')
    all_missing_classes.discard('unknown')
    self.assertEmpty(all_missing_classes)

  def test_namespace_mapping_closure(self):
    # Ensure that all classes in mappings appear in their namespace.
    db = namespace_db.load_db()

    all_missing_classes = set()
    for mapping_name, mapping in db.mappings.items():
      missing_source_classes = set()
      missing_target_classes = set()
      source_namespace = db.namespaces[mapping.source_namespace]
      target_namespace = db.namespaces[mapping.target_namespace]
      for source_cl, target_cl in mapping.mapped_pairs.items():
        if source_cl not in source_namespace.classes:
          missing_source_classes.add(source_cl)
          all_missing_classes.add(source_cl)
        if target_cl not in target_namespace.classes:
          missing_target_classes.add(target_cl)
          all_missing_classes.add(target_cl)
      if missing_source_classes:
        logging.warning(
            'The classes %s in mapping %s did not appear in namespace %s.',
            missing_source_classes,
            mapping_name,
            source_namespace.name,
        )
      if missing_target_classes:
        logging.warning(
            'The classes %s in mapping %s did not appear in namespace %s.',
            missing_target_classes,
            mapping_name,
            target_namespace.name,
        )
      missing_target_classes.discard('unknown')
    self.assertEmpty(all_missing_classes)

  def test_taxonomic_mappings(self):
    # Ensure that all ebird2021 species appear in taxonomic mappings.
    db = namespace_db.load_db()
    ebird = db.namespaces['ebird2021_species']
    genera = db.mappings['ebird2021_to_genus'].mapped_pairs
    families = db.mappings['ebird2021_to_family'].mapped_pairs
    orders = db.mappings['ebird2021_to_order'].mapped_pairs
    missing_genera = set()
    missing_families = set()
    missing_orders = set()
    for cl in ebird.classes:
      if cl not in genera:
        missing_genera.add(cl)
      if cl not in families:
        missing_families.add(cl)
      if cl not in orders:
        missing_orders.add(cl)
    self.assertEmpty(missing_genera)
    self.assertEmpty(missing_families)
    self.assertEmpty(missing_orders)

  def test_namespace_algebra_operators(self):
    ns1 = namespace.Namespace(frozenset(['a', 'b', 'c']))
    ns2 = namespace.Namespace(frozenset(['c', 'd', 'e']))

    # Union
    union_ns = ns1 + ns2
    self.assertEqual(union_ns.classes, frozenset(['a', 'b', 'c', 'd', 'e']))

    # Difference
    diff_ns = ns1 - ns2
    self.assertEqual(diff_ns.classes, frozenset(['a', 'b']))

  def test_algebraic_namespace_notation_and_lookup(self):
    src_path = path_utils.get_absolute_path(
        namespace_db.TAXONOMY_DATABASE_FILENAME
    )
    with tempfile.NamedTemporaryFile(suffix='.sqlite') as tmp:
      shutil.copyfile(src_path, tmp.name)
      db = namespace_db.load_db(path=tmp.name, read_only=False)
      try:
        # Create two temporary namespaces
        db.namespaces['temp_a'] = namespace.Namespace(frozenset(['x', 'y']))
        db.namespaces['temp_b'] = namespace.Namespace(frozenset(['y', 'z']))
        db.namespaces['temp_c'] = namespace.Namespace(frozenset(['x']))

        # Query union A + B
        self.assertIn('temp_a + temp_b', db.namespaces)
        self.assertEqual(
            db.namespaces['temp_a + temp_b'].classes, frozenset(['x', 'y', 'z'])
        )

        # Query difference A - B
        self.assertEqual(
            db.namespaces['temp_a - temp_b'].classes, frozenset(['x'])
        )

        # Query (A + B) - C
        self.assertEqual(
            db.namespaces['(temp_a + temp_b) - temp_c'].classes,
            frozenset(['y', 'z']),
        )
      finally:
        db.conn.close()

  def test_extended_identity_mapping(self):
    src_path = path_utils.get_absolute_path(
        namespace_db.TAXONOMY_DATABASE_FILENAME
    )
    with tempfile.NamedTemporaryFile(suffix='.sqlite') as tmp:
      shutil.copyfile(src_path, tmp.name)
      db = namespace_db.load_db(path=tmp.name, read_only=False)
      try:
        db.namespaces['ns_a'] = namespace.Namespace(
            frozenset(['common1', 'common2', 'only_a'])
        )
        db.namespaces['ns_b'] = namespace.Namespace(
            frozenset(['common1', 'common2', 'only_b'])
        )

        # Mapping with override for common2 and default=True
        mapping = namespace.Mapping(
            source_namespace='ns_a',
            target_namespace='ns_b',
            mapped_pairs={'common2': 'only_b'},
            default=True,
        )
        db.mappings['mapping_a_to_b'] = mapping

        loaded_mapping = db.mappings['mapping_a_to_b']

        # Check default flag
        self.assertTrue(loaded_mapping.default)

        # 'common1' is in both, not overridden -> mapped to 'common1'
        # (default identity behavior)
        self.assertEqual(loaded_mapping.mapped_pairs.get('common1'), 'common1')

        # 'common2' is in both, overridden -> mapped to 'only_b'
        self.assertEqual(loaded_mapping.mapped_pairs.get('common2'), 'only_b')

        # 'only_a' is not in both, not overridden -> not mapped
        self.assertNotIn('only_a', loaded_mapping.mapped_pairs)
      finally:
        db.conn.close()

  def test_sqlite_read_only_mode(self):
    with tempfile.NamedTemporaryFile(suffix='.sqlite') as tmp:
      # Write a small schema and test namespace to the database first
      conn = sqlite3.connect(tmp.name)
      namespace_db.create_tables(conn)
      cursor = conn.cursor()
      cursor.execute(
          'INSERT INTO namespaces (name) VALUES (?)', ('test_read_only_ns',)
      )
      conn.commit()
      conn.close()

      # Now load with read_only=True
      db_ro = namespace_db.load_db(
          path=tmp.name, validate=False, read_only=True
      )
      self.assertIn('test_read_only_ns', db_ro.namespaces)

      # Modifying should raise an error (attempt to write a readonly database)
      with self.assertRaises(sqlite3.OperationalError):
        db_ro.namespaces['new_ns'] = namespace.Namespace(frozenset(['a']))

      # Load with read_only=False
      db_rw = namespace_db.load_db(
          path=tmp.name, validate=False, read_only=False
      )
      db_rw.namespaces['new_ns'] = namespace.Namespace(frozenset(['a']))
      self.assertIn('new_ns', db_rw.namespaces)

  def test_classlist_union_namespace_from_db(self):
    src_path = path_utils.get_absolute_path(
        namespace_db.TAXONOMY_DATABASE_FILENAME
    )
    with tempfile.NamedTemporaryFile(suffix='.sqlite') as tmp:
      shutil.copyfile(src_path, tmp.name)
      db = namespace_db.load_db(path=tmp.name, read_only=False)
      try:
        # Create namespaces in dynamic DB
        db.namespaces['ns_x'] = namespace.Namespace(frozenset(['x1', 'x2']))
        db.namespaces['ns_y'] = namespace.Namespace(frozenset(['y1', 'y2']))

        # ClassList with union namespace name
        union_name = 'ns_x + ns_y'
        cl = namespace.ClassList(namespace=union_name, classes=('x1', 'y2'))

        # Fetch resolved namespace from DB
        db_namespace = db.namespaces[cl.namespace]

        # Verify classes
        self.assertEqual(
            db_namespace.classes, frozenset(['x1', 'x2', 'y1', 'y2'])
        )
        for c in cl.classes:
          self.assertIn(c, db_namespace.classes)
      finally:
        db.conn.close()

  def test_sqlite_in_memory_mode(self):
    with tempfile.NamedTemporaryFile(suffix='.sqlite') as tmp:
      # Write a small schema and test namespace to the database first
      conn = sqlite3.connect(tmp.name)
      namespace_db.create_tables(conn)
      cursor = conn.cursor()
      cursor.execute(
          'INSERT INTO namespaces (name) VALUES (?)', ('test_in_memory_ns',)
      )
      conn.commit()
      conn.close()

      # Load with in_memory=True
      db_mem = namespace_db.load_db(
          path=tmp.name, validate=False, read_only=True, in_memory=True
      )
      self.assertIn('test_in_memory_ns', db_mem.namespaces)

      # Modifying should succeed
      db_mem.namespaces['new_ns'] = namespace.Namespace(frozenset(['a']))
      self.assertIn('new_ns', db_mem.namespaces)

      # Loading without in_memory should see the modifications due to
      # in-memory routing
      db_mem2 = namespace_db.load_db(
          path=tmp.name, validate=False, read_only=True, in_memory=False
      )
      self.assertIn('new_ns', db_mem2.namespaces)


if __name__ == '__main__':
  absltest.main()
