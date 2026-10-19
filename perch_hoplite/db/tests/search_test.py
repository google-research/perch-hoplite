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

"""Tests for Hoplite search."""

import shutil
import tempfile

import numpy as np
from perch_hoplite.db import score_functions
from perch_hoplite.db.tests import test_utils

from absl.testing import absltest
from absl.testing import parameterized

EMBEDDING_SIZE = 128


class SearchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  @parameterized.parameters(*test_utils.DB_TYPES)
  def test_exact_search(self, db_type):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 100, rng, EMBEDDING_SIZE)

    # Pick a random window to search for
    window_ids = db.match_window_ids()
    target_id = window_ids[0]
    target_embedding = db.get_embedding(target_id)

    # Search for the target embedding exactly
    results = db.search(
        target_embedding,
        search_list_size=5,
        approximate=False,
        score_fn_name='cos',
    )

    # We expect the target_id to be one of the top results (score ~1 for cosine)
    found_ids = [r.window_id for r in results]
    self.assertIn(target_id, found_ids)

    # Since we used the exact embedding, the score for target_id should be ~1.0
    for result in results:
      if result.window_id == target_id:
        self.assertAlmostEqual(result.sort_score, 1.0, places=4)

  def test_sqlite_approximate_search(self):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 100, rng, EMBEDDING_SIZE
    )

    # Pick a random window to search for
    window_ids = db.match_window_ids()
    target_id = window_ids[0]
    target_embedding = db.get_embedding(target_id)

    # Search approximately
    results = db.search(target_embedding, search_list_size=5, approximate=True)

    # In such a small DB, approximate search should find the exact match easily
    found_ids = [r.window_id for r in results]
    self.assertIn(target_id, found_ids)

    for r in results:
      if r.window_id == target_id:
        # The score should be high.
        # Whether it's exactly 1.0 depends on normalization.
        self.assertGreater(r.sort_score, 0.1)

  def test_sqlite_approximate_target_score_error(self):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(
        self.tempdir, 'sqlite_usearch', 10, rng, EMBEDDING_SIZE
    )
    query_embedding = rng.normal(size=(EMBEDDING_SIZE,))
    with self.assertRaisesRegex(
        ValueError, 'Approximate search does not support target_score'
    ):
      db.search(
          query_embedding,
          search_list_size=5,
          approximate=True,
          target_score=0.5,
      )

  @parameterized.parameters(*test_utils.DB_TYPES)
  def test_target_score_search(self, db_type):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 100, rng, EMBEDDING_SIZE)

    query_embedding = rng.normal(size=(EMBEDDING_SIZE,))
    # Search for embeddings with score near 0.5
    target_score = 0.5
    results = db.search(
        query_embedding,
        search_list_size=5,
        approximate=False,
        target_score=target_score,
        score_fn_name='cos',
    )
    # The scores should be the actual cosine similarity, because of reranking.
    # So they should be in the range [-1, 1], and likely close to 0.5 if
    # the database is large enough.
    for result in results:
      self.assertGreaterEqual(result.sort_score, -1.0)
      self.assertLessEqual(result.sort_score, 1.0)
      # Since we targeted 0.5, the absolute difference should be small-ish.
      self.assertLess(abs(result.sort_score - target_score), 0.5)

  def test_numpy_neg_euclidean(self):
    rng = np.random.default_rng(42)
    data = rng.normal(scale=2.0, size=(10, EMBEDDING_SIZE))
    queries = rng.normal(scale=2.0, size=(3, EMBEDDING_SIZE))

    # Distance from a vector to itself should be 0.0.
    self.assertAlmostEqual(
        score_functions.numpy_neg_euclidean(queries[0], queries[0]),
        0.0,
        places=5,
    )

    # Single query vs batched data.
    single_scores = score_functions.numpy_neg_euclidean(data, queries[0])
    expected_single = -np.sum((data - queries[0]) ** 2, axis=-1)
    np.testing.assert_allclose(
        single_scores, expected_single, rtol=1e-5, atol=1e-5
    )

    # Multiple queries vs batched data.
    multi_scores = score_functions.numpy_neg_euclidean(data, queries)
    expected_multi = -np.sum(
        (data[:, np.newaxis, :] - queries[np.newaxis, :, :]) ** 2, axis=-1
    )
    np.testing.assert_allclose(
        multi_scores, expected_multi, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        multi_scores[:, 0], single_scores, rtol=1e-5, atol=1e-5
    )

  @parameterized.parameters(*test_utils.DB_TYPES)
  def test_exact_search_neg_euclidean(self, db_type):
    rng = np.random.default_rng(42)
    db = test_utils.make_db(self.tempdir, db_type, 100, rng, EMBEDDING_SIZE)

    window_ids = db.match_window_ids()
    target_id = window_ids[0]
    target_embedding = db.get_embedding(target_id)

    results = db.search(
        target_embedding,
        search_list_size=5,
        approximate=False,
        score_fn_name='neg_euclidean',
    )

    found_ids = [r.window_id for r in results]
    self.assertIn(target_id, found_ids)
    for result in results:
      if result.window_id == target_id:
        self.assertAlmostEqual(result.sort_score, 0.0, places=4)


if __name__ == '__main__':
  absltest.main()
