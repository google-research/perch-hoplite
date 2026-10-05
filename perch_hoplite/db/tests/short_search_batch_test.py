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

"""Regression tests for threaded search batches shorter than top-k."""

import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from perch_hoplite.db import brutalism
from perch_hoplite.db import score_functions
from perch_hoplite.db.tests import test_utils


class ShortSearchBatchTest(parameterized.TestCase):

  @parameterized.product(
      db_type=["in_mem", "sqlite_usearch"],
      case=[
          (1, 1, 4),
          (3, 5, 8),
          (8, 3, 4),
          (9, 3, 4),
          (10, 5, 2),
          (8, 4, 4),
          (9, 1, 4),
      ],
  )
  def test_threaded_matches_exact_search_for_all_partition_sizes(
      self, db_type, case
  ):
    count, top_k, batch_size = case
    with tempfile.TemporaryDirectory() as folder:
      db = test_utils.make_db(
          folder, db_type, count, np.random.default_rng(12), 4
      )
      query = np.array([0.5, -1.0, 0.3, 0.7], dtype=np.float32)
      score_fn = score_functions.get_score_fn("dot")
      before = db.get_embeddings_batch(np.array(db.match_window_ids())).copy()
      expected = list(brutalism.brute_search(db, query, top_k, score_fn))
      for workers in (1, 3):
        actual = list(
            brutalism.threaded_brute_search(
                db,
                query,
                top_k,
                score_fn=score_fn,
                batch_size=batch_size,
                max_workers=workers,
            )
        )
        self.assertLen(actual, min(count, top_k))
        self.assertEqual(
            [r.window_id for r in actual], [r.window_id for r in expected]
        )
        np.testing.assert_allclose(
            [r.sort_score for r in actual],
            [r.sort_score for r in expected],
            rtol=1e-6,
        )
      np.testing.assert_array_equal(
          db.get_embeddings_batch(np.array(db.match_window_ids())), before
      )

  @parameterized.parameters(2, 0.25)
  def test_sampled_search_can_return_fewer_than_requested(self, sample_size):
    with tempfile.TemporaryDirectory() as folder:
      db = test_utils.make_db(folder, "in_mem", 8, np.random.default_rng(13), 4)
      query = np.ones(4, dtype=np.float32)
      score_fn = score_functions.get_score_fn("cos")
      expected = list(
          brutalism.brute_search(
              db, query, 5, score_fn, sample_size=sample_size, rng_seed=3
          )
      )
      actual = list(
          brutalism.threaded_brute_search(
              db,
              query,
              5,
              score_fn,
              sample_size=sample_size,
              rng_seed=3,
              batch_size=4,
          )
      )
      self.assertEqual(
          [r.window_id for r in actual], [r.window_id for r in expected]
      )
      self.assertLen(actual, 2)

  def test_empty_database_returns_no_matches(self):
    with tempfile.TemporaryDirectory() as folder:
      db = test_utils.make_db(folder, "in_mem", 0, np.random.default_rng(1), 4)
      result = brutalism.threaded_brute_search(
          db, np.ones(4), 5, score_fn=score_functions.get_score_fn("dot")
      )
      self.assertEmpty(list(result))

  def test_public_database_search_handles_short_final_batch(self):
    with tempfile.TemporaryDirectory() as folder:
      db = test_utils.make_db(folder, "in_mem", 9, np.random.default_rng(9), 4)
      ids = db.match_window_ids()
      query = db.get_embedding(ids[-1])
      actual = db.search(
          query,
          search_list_size=3,
          approximate=False,
          score_fn_name="cos",
          batch_size=4,
      )
      self.assertIn(ids[-1], [r.window_id for r in actual])
      self.assertLen(list(actual), 3)


if __name__ == "__main__":
  absltest.main()
