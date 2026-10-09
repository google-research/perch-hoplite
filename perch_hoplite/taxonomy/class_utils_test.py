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
"""Tests for automatic resolution of species with unknown source namespaces."""

import unittest

from perch_hoplite.taxonomy import class_utils
from perch_hoplite.taxonomy import namespace
from perch_hoplite.taxonomy import namespace_db


class SpeciesConversionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    namespaces = {
        'inat2024': namespace.Namespace(
            frozenset(
                [
                    'american_robin',
                    'sparrow',
                    'owl',
                    'cougar',
                    'same',
                    'shared_alias',
                ]
            )
        ),
        'ebird': namespace.Namespace(
            frozenset(
                ['amerob', 'dup', 'ambiguous', 'no_map', 'same', 'shared_alias']
            )
        ),
        'aou': namespace.Namespace(frozenset(['amerob', 'dup', 'ambiguous'])),
        'other': namespace.Namespace(frozenset(['not_converted'])),
    }
    mappings = {
        'ebird_to_inat2024': namespace.Mapping(
            source_namespace='ebird',
            target_namespace='inat2024',
            mapped_pairs={
                'amerob': 'american_robin',
                'dup': 'sparrow',
                'ambiguous': 'owl',
                'shared_alias': 'owl',
            },
            default=True,
        ),
        'aou_to_inat2024': namespace.Mapping(
            source_namespace='aou',
            target_namespace='inat2024',
            mapped_pairs={
                'amerob': 'american_robin',
                'dup': 'sparrow',
                'ambiguous': 'cougar',
            },
        ),
    }
    self.db = namespace_db.TaxonomyDatabase(namespaces, mappings=mappings)
    self.addCleanup(self.db.conn.close)

  def convert(self, species, **kwargs):
    return class_utils.convert_species_to_namespace(
        species, taxonomy_database=self.db, **kwargs
    )

  def test_single_unknown_provenance_species(self):
    self.assertEqual(self.convert('amerob'), 'american_robin')

  def test_identical_mapping_from_two_possible_sources_is_not_ambiguous(self):
    self.assertEqual(self.convert('dup'), 'sparrow')

  def test_multiple_species_preserves_order_and_duplicates(self):
    self.assertEqual(
        self.convert(['dup', 'amerob', 'dup', 'same']),
        ['sparrow', 'american_robin', 'sparrow', 'same'],
    )

  def test_tuple_input_returns_list(self):
    self.assertEqual(
        self.convert(('amerob', 'dup')), ['american_robin', 'sparrow']
    )

  def test_direct_target_namespace_label_is_unchanged(self):
    self.assertEqual(self.convert('american_robin'), 'american_robin')

  def test_default_identity_mapping_fills_shared_species(self):
    self.assertEqual(self.convert('same', source_namespace='ebird'), 'same')

  def test_conflicting_source_mappings_fail_by_default(self):
    with self.assertRaisesRegex(ValueError, 'Ambiguous species'):
      self.convert('ambiguous')

  def test_known_source_disambiguates_conflicting_species(self):
    self.assertEqual(self.convert('ambiguous', source_namespace='ebird'), 'owl')
    self.assertEqual(
        self.convert('ambiguous', source_namespace='aou'), 'cougar'
    )

  def test_shared_canonical_name_with_override_is_not_silently_guessed(self):
    with self.assertRaisesRegex(ValueError, 'Ambiguous species'):
      self.convert('shared_alias')
    self.assertEqual(
        self.convert('shared_alias', source_namespace='ebird'), 'owl'
    )
    self.assertEqual(
        self.convert('shared_alias', source_namespace='inat2024'),
        'shared_alias',
    )

  def test_missing_species_raises_by_default(self):
    with self.assertRaisesRegex(ValueError, 'No mapping'):
      self.convert('not_a_real_species')

  def test_no_conversion_path_for_known_source_raises(self):
    with self.assertRaisesRegex(ValueError, 'No mapping'):
      self.convert('no_map', source_namespace='ebird')

  def test_warn_mode_logs_and_retains_batch_positions(self):
    with self.assertLogs(level='WARNING') as logged:
      result = self.convert(
          ['amerob', 'missing', 'ambiguous', 'dup'], errors='warn'
      )
    self.assertEqual(result, ['american_robin', None, None, 'sparrow'])
    self.assertEqual(len(logged.output), 2)
    self.assertIn('No mapping', logged.output[0])
    self.assertIn('Ambiguous species', logged.output[1])

  def test_ignore_mode_returns_none_without_logging(self):
    with self.assertNoLogs(level='WARNING'):
      self.assertEqual(
          self.convert(['not_a_real_species', 'ambiguous'], errors='ignore'),
          [None, None],
      )

  def test_single_missing_species_returns_none_when_ignored(self):
    self.assertIsNone(self.convert('absent', errors='ignore'))

  def test_reject_unknown_namespace(self):
    with self.assertRaisesRegex(ValueError, 'Unknown target namespace'):
      self.convert('amerob', target_namespace='unknown')
    with self.assertRaisesRegex(ValueError, 'Unknown source namespace'):
      self.convert('amerob', source_namespace='unknown')

  def test_reject_invalid_error_policy(self):
    with self.assertRaisesRegex(ValueError, 'errors must be'):
      self.convert('amerob', errors='surprise')

  def test_reject_invalid_types(self):
    with self.assertRaises(TypeError):
      self.convert(42)
    with self.assertRaises(TypeError):
      self.convert(['amerob', 42])

  def test_empty_batch_returns_empty_list(self):
    self.assertEqual(self.convert([]), [])

  def test_real_taxonomy_database_mapping(self):
    real_db = namespace_db.load_db()
    source = 'aou_bird_codes'
    mapping = real_db.mappings['aou_bird_codes_to_inat2024']
    label, expected = next(
        (label, target)
        for label, target in mapping.mapped_pairs.items()
        if label in real_db.namespaces[source].classes
        and target in real_db.namespaces['inat2024'].classes
    )
    self.assertEqual(
        class_utils.convert_species_to_namespace(
            label,
            source_namespace=source,
            taxonomy_database=real_db,
        ),
        expected,
    )


if __name__ == '__main__':
  unittest.main()
