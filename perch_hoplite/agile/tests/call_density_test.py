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

"""Tests for call density estimation."""

import numpy as np
from perch_hoplite.agile import call_density
from perch_hoplite.agile.tests import test_utils

from absl.testing import absltest


class CallDensityTest(absltest.TestCase):

  def test_atb_end_to_end(self):
    # Create the test database and ground truth labels.
    # pos_pi is the groundtruth call density.
    pos_pi = 0.07
    db, linear_classifier, gt_labels = test_utils.call_density_simulation(
        num_windows=8192,
        embedding_dim=2,
        rng_seed=42,
        mu_diff=2.0,
        pos_sigma=1.0,
        neg_sigma=1.0,
        pos_pi=pos_pi,
    )

    # Create the CallDensityATB object.
    cde_atb = call_density.CallDensityATB.create(
        db,
        42,
        study_name='test_run',
        classifier=linear_classifier,
        target_class='test_class',
        bin_bounds=[0.0, 0.5, 0.75, 0.875, 1.0],
        samples_per_bin=100,
        binning_strategy='quantile',
    )

    # Insert validation labels.
    validation_examples = cde_atb.get_validation_examples()
    for ex in validation_examples:
      window = db.get_window(ex.window_id)
      db.insert_annotation(
          window.recording_id,
          window.offsets,
          'test_class',
          gt_labels[ex.window_id],
          'cde_validation',
          handle_duplicates='allow',
      )

    # Estimate call density.
    mean_estimate, sample_estimates = cde_atb.estimate_call_density(db)
    low, high = np.quantile(sample_estimates, [0.05, 0.95])
    self.assertAlmostEqual(mean_estimate, pos_pi, delta=0.05)
    self.assertGreater(high, pos_pi)
    self.assertLess(low, pos_pi)

    # Serialize and deserialize the CallDensityATB object.
    # config_dict = cde_atb.to_config_dict()
    cde_atb.save_to_db(db)
    cde_atb_from_config = call_density.CallDensityATB.load_from_db(
        db, 'test_run', 'test_class'
    )
    self.assertEqual(cde_atb.study_name, cde_atb_from_config.study_name)
    self.assertEqual(cde_atb.target_class, cde_atb_from_config.target_class)
    np.testing.assert_allclose(
        cde_atb.classifier.beta, cde_atb_from_config.classifier.beta
    )
    np.testing.assert_allclose(
        cde_atb.classifier.beta_bias, cde_atb_from_config.classifier.beta_bias
    )
    self.assertEqual(
        cde_atb.samples_per_bin, cde_atb_from_config.samples_per_bin
    )
    for i in range(5):
      self.assertEqual(
          cde_atb.validation_examples.examples[i].window_id,
          cde_atb_from_config.validation_examples.examples[i].window_id,
      )

    # Check that the reconstructed estimator gives the same result.
    mean_estimate_2, _ = cde_atb_from_config.estimate_call_density(db)
    self.assertAlmostEqual(mean_estimate_2, mean_estimate)


if __name__ == '__main__':
  absltest.main()
