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

"""Euclidean score and nearest-neighbor regression tests."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from perch_hoplite.db import brutalism
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import score_functions


class EuclideanScoreTest(parameterized.TestCase):

  @parameterized.product(dtype=(np.float32, np.float64), batched=(False, True))
  def test_scores_are_negative_distances(self, dtype, batched):
    data = np.array(
        [[1.0, 0.0], [1.4, 0.0], [10.0, 0.0], [0.0, 0.0], [-2.0, 3.0]],
        dtype=dtype,
    )
    queries = np.array([[1.0, 0.0], [-1.0, 2.0]], dtype=dtype)
    query = queries if batched else queries[0]
    expected = -np.linalg.norm(data[:, None, :] - queries, axis=-1)
    if not batched:
      expected = expected[:, 0]
    actual = score_functions.numpy_neg_euclidean(data, query)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    self.assertTrue(np.all(actual <= 0))
    self.assertEqual(np.shape(actual), expected.shape)

  @parameterized.parameters(False, True)
  def test_single_data_vector_matches_row_in_a_batch(self, batched_query):
    data = np.array([3.0, 4.0])
    queries = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    query = queries if batched_query else queries[0]
    expected = -np.linalg.norm(data - query, axis=-1)
    np.testing.assert_allclose(
        score_functions.numpy_neg_euclidean(data, query), expected
    )

  @parameterized.parameters(np.float32, np.float64)
  def test_nearby_large_vectors_preserve_translation_invariance(self, dtype):
    data = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=dtype)
    query = np.zeros((2,), dtype=dtype)
    for batched in (False, True):
      q = query[None] if batched else query
      expected = score_functions.numpy_neg_euclidean(data, q)
      actual = score_functions.numpy_neg_euclidean(data + 1e6, q + 1e6)
      np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
      np.testing.assert_allclose(actual.reshape(-1), [-5.0, -13.0], rtol=1e-6)

  def test_many_queries_match_individual_query_scores(self):
    rng = np.random.default_rng(42)
    data, queries = rng.normal(size=(5, 4)), rng.normal(size=(3, 4))
    expected = np.stack(
        [score_functions.numpy_neg_euclidean(data, q) for q in queries], axis=-1
    )
    np.testing.assert_allclose(
        score_functions.numpy_neg_euclidean(data, queries), expected, rtol=1e-12
    )

  def test_empty_batches_preserve_pairwise_shapes(self):
    self.assertEqual(
        score_functions.numpy_neg_euclidean(
            np.empty((0, 3)), np.zeros(3)
        ).shape,
        (0,),
    )
    self.assertEqual(
        score_functions.numpy_neg_euclidean(
            np.empty((0, 3)), np.zeros((2, 3))
        ).shape,
        (0, 2),
    )
    self.assertEqual(
        score_functions.numpy_neg_euclidean(
            np.zeros((2, 3)), np.empty((0, 3))
        ).shape,
        (2, 0),
    )

  def test_bias_and_target_wrappers_use_true_distance(self):
    data = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    fn = score_functions.get_score_fn(
        "neg_euclidean", bias=0.5, target_score=-1.5
    )
    expected = -np.abs(np.array([0.0, -5.0, -1.0]) + 2.0)
    np.testing.assert_allclose(fn(data, np.zeros(2)), expected)

  def test_real_database_search_returns_the_closest_embedding(self):
    db = in_mem_impl.InMemoryGraphSearchDB.create(
        embedding_dim=2, embedding_dtype=np.float32
    )
    rid = db.insert_recording("test.wav")
    data = np.array(
        [[1.0, 0.0], [1.4, 0.0], [10.0, 0.0], [0.0, 0.0], [-2.0, 0.0]],
        dtype=np.float32,
    )
    ids = [
        db.insert_window(rid, [float(i), float(i + 1)], x)
        for i, x in enumerate(data)
    ]
    query = np.array([1.0, 0.0], dtype=np.float32)
    score_fn = score_functions.get_score_fn("neg_euclidean")
    direct = brutalism.brute_search(
        db, query, search_list_size=2, score_fn=score_fn
    )
    threaded = db.search(
        query,
        search_list_size=2,
        approximate=False,
        score_fn_name="neg_euclidean",
        max_workers=1,
    )
    for results in (direct, threaded):
      ranked = list(results)
      self.assertEqual([result.window_id for result in ranked], ids[:2])
      np.testing.assert_allclose(
          [r.sort_score for r in ranked], [0.0, -0.4], atol=1e-6
      )


if __name__ == "__main__":
  absltest.main()
