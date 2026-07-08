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
from perch_hoplite.db import datatypes

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
    validation_examples = cde_atb.validation_examples
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

  def test_platt_scaling_end_to_end(self):
    # Create the test database and ground truth labels.
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

    # Create the CallDensityPlattScaling object.
    cde_platt = call_density.CallDensityPlattScaling.create(
        db,
        42,
        study_name='test_run_platt',
        classifier=linear_classifier,
        target_class='test_class',
        sampling_exponent=3.0,
        sampling_epsilon=0.05,
        n_validation_samples=200,
    )

    # Insert validation labels.
    validation_examples = cde_platt.validation_examples
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
    mean_estimate, sample_estimates = cde_platt.estimate_call_density(
        db, rng_seed=42
    )
    low, high = np.quantile(sample_estimates, [0.05, 0.95])
    self.assertAlmostEqual(mean_estimate, pos_pi, delta=0.05)
    self.assertGreater(high, pos_pi)
    self.assertLess(low, pos_pi)
    self.assertGreater(cde_platt.estimate_roc_auc(), 0.8)

    # Serialize and deserialize the CallDensityPlattScaling object.
    cde_platt.save_to_db(db)
    cde_platt_from_config = call_density.CallDensityPlattScaling.load_from_db(
        db, 'test_run_platt', 'test_class'
    )
    self.assertEqual(cde_platt.study_name, cde_platt_from_config.study_name)
    self.assertEqual(cde_platt.target_class, cde_platt_from_config.target_class)
    np.testing.assert_allclose(
        cde_platt.classifier.beta, cde_platt_from_config.classifier.beta
    )
    np.testing.assert_allclose(
        cde_platt.classifier.beta_bias,
        cde_platt_from_config.classifier.beta_bias,
    )
    self.assertEqual(
        cde_platt.sampling_exponent, cde_platt_from_config.sampling_exponent
    )
    for i in range(5):
      self.assertEqual(
          cde_platt.validation_examples.examples[i].window_id,
          cde_platt_from_config.validation_examples.examples[i].window_id,
      )

    mean_estimate_2, _ = cde_platt_from_config.estimate_call_density(
        db, rng_seed=42
    )
    self.assertAlmostEqual(mean_estimate_2, mean_estimate)

  def test_platt_deployment_density(self):
    # Create the test database and ground truth labels.
    db, linear_classifier, gt_labels = test_utils.call_density_simulation(
        num_windows=100, pos_pi=0.5
    )
    # The simulation creates a deployment named 'test' with project 'test'.

    # Add a second deployment with low density.
    did2 = db.insert_deployment(name='test2', project='test')
    rid2 = db.insert_recording(filename='test2', deployment_id=did2)
    for i in range(10):
      # Very negative score to ensure low probability.
      embedding = -10.0 * linear_classifier.beta.flatten()
      wid = db.insert_window(
          rid2,
          [500.0 + 5.0 * i, 500.0 + 5.0 * i + 5.0],
          embedding=embedding,
      )
      gt_labels[wid] = datatypes.LabelType.NEGATIVE

    # Create the CallDensityPlattScaling object.
    cde_platt = call_density.CallDensityPlattScaling.create(
        db,
        42,
        'test_study',
        linear_classifier,
        'test_class',
        sampling_exponent=1.0,
        sampling_epsilon=0.0,
        n_validation_samples=100,
    )

    # Add validation labels
    for ex in cde_platt.validation_examples:
      window = db.get_window(ex.window_id)
      db.insert_annotation(
          window.recording_id,
          window.offsets,
          'test_class',
          gt_labels[ex.window_id],
          'test',
      )

    # Estimate
    results = cde_platt.estimate_deployment_call_density(db, n_resamples=10)
    did1 = next(d.id for d in db.get_all_deployments() if d.name == 'test')
    self.assertIn(did1, results)
    self.assertIn(did2, results)
    mle, samples = results[did1]
    mle2, samples2 = results[did2]
    # 100 windows, 0.5 pi -> around 0.5 density.
    self.assertLess(mle, 1.0)
    self.assertGreater(mle, 0.1)
    self.assertGreater(mle, mle2)
    self.assertLen(samples, 10)
    self.assertLen(samples2, 10)


if __name__ == '__main__':
  absltest.main()
