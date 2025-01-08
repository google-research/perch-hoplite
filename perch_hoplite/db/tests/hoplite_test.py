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

"""Tests for Hoplite databases."""

import shutil
import tempfile

import numpy as np
from perch_hoplite.db import brutalism
from perch_hoplite.db import interface
from perch_hoplite.db.tests import test_utils

from absl.testing import absltest
from absl.testing import parameterized

EMBEDDING_SIZE = 128
DB_TYPES = ('in_mem', 'sqlite_usearch')
DB_TYPE_NAMED_PAIRS = (
    ('in_mem-sqlite_usearch', 'in_mem', 'sqlite_usearch'),
)
PERSISTENT_DB_TYPES = ('sqlite_usearch',)


class HopliteTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  @parameterized.product(
      db_type=DB_TYPES,
      thread_split=(True, False),
  )
  def test_graph_db_interface(self, db_type, thread_split):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 1000, rng, EMBEDDING_SIZE)
    if thread_split:
      db = db.thread_split()

    # Run all methods in the interface...
    self.assertEqual(db.count_embeddings(), 1000)
    idxes = db.get_embedding_ids()

    one_idx = db.get_one_embedding_id()
    self.assertIn(one_idx, idxes)

    # Check the metadata.
    got_md = db.get_metadata('db_config')
    self.assertEqual(got_md.embedding_dim, EMBEDDING_SIZE)

    with self.subTest('test_embedding_sources'):
      source = db.get_embedding_source(idxes[1])
      # The embeddings are given one of three randomly selected dataset names.
      embs = db.get_embeddings_by_source(source.dataset_name, None, None)
      self.assertGreater(embs.shape[0], db.count_embeddings() / 6)
      # For an unknown dataset name, we should get no embeddings.
      embs = db.get_embeddings_by_source('fake_name', None, None)
      self.assertEqual(embs.shape[0], 0)
      # Source ids are approximately unique.
      embs = db.get_embeddings_by_source(
          source.dataset_name, source.source_id, None
      )
      self.assertLen(embs, 1)
      # For an unknown source id, we should get no embeddings.
      embs = db.get_embeddings_by_source(source.dataset_name, 'fake_id', None)
      self.assertEqual(embs.shape[0], 0)

      sources = db.get_embedding_sources(idxes[:3])
      self.assertLen(sources, 3)
    db.commit()

  @parameterized.product(db_type=PERSISTENT_DB_TYPES)
  def test_persistence(self, db_type):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 1000, rng, EMBEDDING_SIZE)
    one_emb = np.random.normal(size=(EMBEDDING_SIZE,), loc=0, scale=0.05)
    one_emb_id = db.insert_embedding(
        one_emb, source=interface.EmbeddingSource('q', 'x', np.array([5.0]))
    )
    self.assertEqual(db.get_embedding_ids().shape[0], 1001)
    db.commit()

    got_emb = db.get_embedding(one_emb_id)
    np.testing.assert_equal(got_emb, np.float16(one_emb))

    # "Making" the persistent DB without adding any new embeddings gives us a
    # view of the saved DB.
    test_db = test_utils.make_db(self.tempdir, db_type, 0, rng, EMBEDDING_SIZE)
    self.assertIn(one_emb_id, test_db.get_embedding_ids())
    # Check that the embeddings are the same in the two DB's.
    for idx in db.get_embedding_ids():
      emb = db.get_embedding(idx)
      test_emb = test_db.get_embedding(idx)
      np.testing.assert_equal(emb, test_emb)

  @parameterized.product(db_type=DB_TYPES)
  def test_labels_db_interface(self, db_type):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 1000, rng, EMBEDDING_SIZE)
    ids = db.get_embedding_ids()
    db.insert_label(
        interface.Label(ids[0], 'hawgoo', interface.LabelType.POSITIVE, 'human')
    )
    db.insert_label(
        interface.Label(
            ids[0], 'hawgoo', interface.LabelType.POSITIVE, 'machine'
        )
    )
    db.insert_label(
        interface.Label(
            ids[1], 'hawgoo', interface.LabelType.POSITIVE, 'machine'
        )
    )
    db.insert_label(
        interface.Label(
            ids[0], 'rewbla', interface.LabelType.NEGATIVE, 'machine'
        )
    )

    with self.subTest('get_embeddings_by_label'):
      # When both label_type and provenance are unspecified, we should get all
      # unique IDs with the target label. Id's 0 and 1 both have some kind of
      # 'hawgoo' label.
      got = db.get_embeddings_by_label('hawgoo', None, None)
      self.assertSequenceEqual(sorted(got), sorted([ids[0], ids[1]]))

    with self.subTest('get_embeddings_by_label_type'):
      # Now we should get the ID's for all POSITIVE 'hawgoo' labels, regardless
      # of provenance.
      got = db.get_embeddings_by_label(
          'hawgoo', interface.LabelType.POSITIVE, None
      )
      self.assertSequenceEqual(sorted(got), sorted([ids[0], ids[1]]))

      # There are no negative 'hawgoo' labels.
      got = db.get_embeddings_by_label(
          'hawgoo', interface.LabelType.NEGATIVE, None
      )
      self.assertEqual(got.shape[0], 0)

    with self.subTest('get_embeddings_by_label_provenance'):
      # There is only one hawgoo labeled by a human.
      got = db.get_embeddings_by_label('hawgoo', None, 'human')
      self.assertSequenceEqual(got, [ids[0]])

      # And only one example with a 'rewbla' labeled by a machine.
      got = db.get_embeddings_by_label('rewbla', None, 'machine')
      self.assertSequenceEqual(got, [ids[0]])

    with self.subTest('count_all_labels'):
      # Finally, there are a total of three labels on ID 0.
      got = db.get_labels(ids[0])
      self.assertLen(got, 3)

    with self.subTest('get_classes'):
      got = db.get_classes()
      self.assertSequenceEqual(got, ['hawgoo', 'rewbla'])

    with self.subTest('get_class_counts'):
      # 2 positive labels for 'hawgoo' ignoring provenance, 0 for 'rewbla'.
      got = db.get_class_counts(interface.LabelType.POSITIVE)
      self.assertDictEqual(got, {'hawgoo': 2, 'rewbla': 0})

      # 1 negative label for 'rewbla', 0 for 'hawgoo'.
      got = db.get_class_counts(interface.LabelType.NEGATIVE)
      self.assertDictEqual(got, {'hawgoo': 0, 'rewbla': 1})

    with self.subTest('count_classes'):
      self.assertEqual(db.count_classes(), 2)

    with self.subTest('duplicate_labels'):
      dupe_label = interface.Label(
          ids[0], 'hawgoo', interface.LabelType.POSITIVE, 'human'
      )
      self.assertFalse(db.insert_label(dupe_label, skip_duplicates=True))
      self.assertTrue(db.insert_label(dupe_label, skip_duplicates=False))

  @parameterized.named_parameters(*DB_TYPE_NAMED_PAIRS)
  def test_brute_search_impl_agreement(self, target_db_type, source_db_type):
    rng = np.random.default_rng(42)
    source_db = test_utils.make_db(
        self.tempdir, source_db_type, 1000, rng, EMBEDDING_SIZE
    )
    target_db = test_utils.make_db(
        self.tempdir,
        target_db_type,
        1000,
        rng,
        EMBEDDING_SIZE,
        fill_random=False,
    )
    id_mapping = test_utils.clone_embeddings(source_db, target_db)

    # Check brute-force search agreement.
    query = rng.normal(size=(128,), loc=0, scale=1.0)
    results_m, _ = brutalism.brute_search(
        source_db, query, search_list_size=10, score_fn=np.dot
    )
    results_s, _ = brutalism.brute_search(
        target_db, query, search_list_size=10, score_fn=np.dot
    )
    self.assertLen(results_m.search_results, 10)
    self.assertLen(results_s.search_results, 10)
    # Search results are iterated over in sorted order.
    for r_m, r_s in zip(results_m, results_s):
      emb_m = source_db.get_embedding(r_m.embedding_id)
      emb_s = target_db.get_embedding(r_s.embedding_id)
      self.assertEqual(id_mapping[r_m.embedding_id], r_s.embedding_id)
      # TODO(tomdenton): check that the scores are the same.
      np.testing.assert_equal(emb_m, emb_s)

  def test_random_batched_iterator(self):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, 'in_mem', 100, rng, EMBEDDING_SIZE)
    # Test behavior when batch size divides the number of embeddings.
    with self.subTest('batch_size_divides_num_embeddings'):
      batch_size = 10
      iterator = db.random_batched_iterator(
          db.get_embedding_ids(), batch_size=batch_size, rng=rng
      )
      all_ids = set()
      for i in range(10):
        batch = next(iterator)
        self.assertLen(batch, batch_size)
        self.assertContainsSubset(batch, db.get_embedding_ids())
        all_ids.update(batch)
        self.assertLen(all_ids, min(100, (i + 1) * batch_size))

    # Test behavior when batch size does not divide the number of embeddings.
    with self.subTest('batch_size_does_not_divide_num_embeddings'):
      batch_size = 27
      iterator = db.random_batched_iterator(
          db.get_embedding_ids(), batch_size=batch_size, rng=rng
      )
      all_ids = set()
      for i in range(4):
        batch = next(iterator)
        self.assertLen(batch, batch_size)
        all_ids.update(batch)
        self.assertContainsSubset(batch, db.get_embedding_ids())
        self.assertLen(all_ids, min(100, (i + 1) * batch_size))


if __name__ == '__main__':
  absltest.main()
