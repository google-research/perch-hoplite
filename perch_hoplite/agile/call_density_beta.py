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

"""Mixture modeling for call density estimation."""

import dataclasses

from ml_collections import config_dict
import numpy as np
from perch_hoplite.agile import call_density
from perch_hoplite.agile import classifier as classifier_lib
from perch_hoplite.db import datatypes
from perch_hoplite.db import interface
from scipy import stats


@dataclasses.dataclass(kw_only=True)
class CallDensityWeightedBeta(call_density.CallDensityEstimator):
  """Call density estimation using a sample-weighted Beta distribution."""

  sampling_exponent: float
  sampling_epsilon: float

  # Filters used for selecting windows for density estimation.
  deployments_filter: config_dict.ConfigDict | None = None
  recordings_filter: config_dict.ConfigDict | None = None

  @classmethod
  def cde_method(cls) -> str:
    return 'mixture'

  @classmethod
  def create(
      cls,
      db: interface.HopliteDBInterface,
      rng_seed: int,
      study_name: str,
      classifier: classifier_lib.LinearClassifier,
      target_class: str,
      sampling_exponent: float,
      sampling_epsilon: float,
      n_validation_samples: int,
      deterministic: bool = True,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      binning_sample_size: int = 0,
  ):
    """Creates a CallDensityMixture object."""
    window_ids, logits = call_density.match_windows_and_compute_logits(
        db,
        rng_seed,
        classifier,
        target_class,
        deployments_filter,
        recordings_filter,
        sample_size=binning_sample_size,
    )
    valid_window_ids, valid_logits, sample_weights = (
        call_density.monomial_sample_selection(
            window_ids,
            logits,
            n_validation_samples,
            sampling_exponent,
            sampling_epsilon,
            deterministic=deterministic,
            rng_seed=rng_seed,
        )
    )
    examples = []
    for window_id, logit, sample_weight in zip(
        valid_window_ids, valid_logits, sample_weights
    ):
      examples.append(
          call_density.ValidationExample(
              window_id=int(window_id),
              label_type=None,
              score=float(logit),
              sample_weight=float(sample_weight),
          )
      )
    examples = call_density.ValidationSet(examples=examples)
    examples.update_annotations_from_db(db, target_class)
    return cls(
        study_name=study_name,
        classifier=classifier,
        target_class=target_class,
        validation_examples=examples,
        sampling_exponent=sampling_exponent,
        sampling_epsilon=sampling_epsilon,
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
    )

  def estimate_call_density(
      self,
      db: interface.HopliteDBInterface,
      rng_seed: int = 42,
      scores_sample_size: int = 10000,
      n_resamples: int = 1024,
      default_label_type: datatypes.LabelType = datatypes.LabelType.NEGATIVE,
      **kwargs,
  ) -> tuple[float, np.ndarray]:
    del scores_sample_size  # Unused in weighted beta approach.
    self.validation_examples.update_annotations_from_db(db, self.target_class)

    # Fill in None and UNCERTAIN if desired.
    # Note: validation_examples is mutated here, consistent with other methods.
    for ex in self.validation_examples:
      if ex.label_type is None:
        ex.label_type = default_label_type

    def get_weighted_counts(examples):
      p_sum = sum(
          e.sample_weight
          for e in examples
          if e.label_type == datatypes.LabelType.POSITIVE
      )
      n_sum = sum(
          e.sample_weight
          for e in examples
          if e.label_type == datatypes.LabelType.NEGATIVE
      )
      return p_sum, n_sum

    pos_sum, neg_sum = get_weighted_counts(self.validation_examples.examples)
    total_sum = pos_sum + neg_sum + 1e-9
    num_points = len(self.validation_examples.examples)

    # Scale weights so they sum to the actual sample size num_points.
    # This ensures the Beta distribution variance reflects the actual labels.
    scale = num_points / total_sum
    beta_prior = kwargs.get('beta_prior', 0.1)

    def get_beta_mle(p_s, n_s, s, prior):
      alpha = p_s * s + prior
      beta = n_s * s + prior
      if alpha > 1 and beta > 1:
        return (alpha - 1) / (alpha + beta - 2)
      else:
        return alpha / (alpha + beta)

    p_pos_mle = get_beta_mle(pos_sum, neg_sum, scale, beta_prior)

    # Produce candidate scores by bootstrap resampling with Beta sampling.
    bootstrap_p_pos = []
    rng = np.random.default_rng(rng_seed)
    for _ in range(n_resamples):
      resampled = self.validation_examples.bootstrap_sample(rng)
      r_p_sum, r_n_sum = get_weighted_counts(resampled.examples)
      r_total = r_p_sum + r_n_sum + 1e-9
      # Re-scale for each resample to maintain consistent effective sample size.
      r_scale = num_points / r_total
      alpha = r_p_sum * r_scale + beta_prior
      beta = r_n_sum * r_scale + beta_prior
      # Sample from the Beta distribution to capture posterior uncertainty.
      bootstrap_p_pos.append(stats.beta(alpha, beta).rvs(random_state=rng))

    return p_pos_mle, np.array(bootstrap_p_pos)

  def estimate_deployment_call_density(
      self,
      db: interface.HopliteDBInterface,
      **kwargs,
  ) -> dict[str, tuple[float, np.ndarray]]:
    """Estimates call density per deployment."""
    # TODO(tomdenton): Implement per-deployment call density estimation.
    raise NotImplementedError

  def estimate_roc_auc(self) -> float:
    pos_scores = np.array([
        e.score
        for e in self.validation_examples.examples
        if e.label_type == datatypes.LabelType.POSITIVE
    ])
    neg_scores = np.array([
        e.score
        for e in self.validation_examples.examples
        if e.label_type == datatypes.LabelType.NEGATIVE
    ])
    if pos_scores.size == 0 or neg_scores.size == 0:
      return 0.0
    hits = 0
    for ps in pos_scores:
      hits += (ps > neg_scores).sum()
      hits += (ps == neg_scores).sum() * 0.5
    roc_auc = hits / (pos_scores.size * neg_scores.size)
    return roc_auc
