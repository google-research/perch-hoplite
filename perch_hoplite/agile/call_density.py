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

"""Tooling for measuring call density."""

import collections
import dataclasses
from typing import Sequence

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import interface
from perch_hoplite.db import search_results
import scipy


@dataclasses.dataclass
class ValidationExample:
  """Wrapper for validation data used for call density estimation.

  Attributes:
    window_id: Window ID corresponding to the example.
    label_type: Label type (positive, negative or uncertain). Can be None if the
      example is unlabeled.
    score: Classifier score.
    bin: Bin number the validation example was assigned to.
    bin_weight: Proportion of total data in the same bin as this example.
  """

  window_id: int
  label_type: interface.LabelType | None
  score: float
  bin: int
  bin_weight: float


@dataclasses.dataclass
class CallDensityConfig(interface.HopliteConfig):
  """Config dataclass for call density estimation."""

  study_name: str
  classifier: config_dict.ConfigDict
  target_class: str
  quantile_bounds: list[float]
  samples_per_bin: int

  window_ids: dataclasses.InitVar[list[int]]
  logits: dataclasses.InitVar[list[float]]
  annotations: dataclasses.InitVar[list[interface.Annotation | None] | None] = (
      None
  )

  # TODO(stefanistrate): Allow setting the value bounds directly (instead of
  # computing them from the quantile bounds).
  value_bounds: list[float] = dataclasses.field(init=False)
  bins_dict: dict[int, list[ValidationExample]] = dataclasses.field(init=False)

  deployments_filter: config_dict.ConfigDict | None = None
  recordings_filter: config_dict.ConfigDict | None = None
  windows_filter: config_dict.ConfigDict | None = None
  annotations_filter: config_dict.ConfigDict | None = None

  def __post_init__(
      self,
      window_ids: list[int],
      logits: list[float],
      annotations: list[interface.Annotation | None] | None,
  ) -> None:
    """Post-init initialization."""

    if annotations is None:
      annotations = [None] * len(window_ids)

    self.value_bounds = np.quantile(logits, self.quantile_bounds).tolist()

    bin_numbers = np.digitize(logits, self.value_bounds)
    bin_weights = np.diff(self.quantile_bounds, prepend=-1.0)

    self.bins_dict = {bin: [] for bin in range(1, len(self.value_bounds))}
    for bin_number, window_id, logit, annotation in zip(
        bin_numbers, window_ids, logits, annotations
    ):
      if bin_number == 0:
        bin_number = 1
      elif bin_number == len(self.value_bounds):
        bin_number -= 1
      self.bins_dict[bin_number].append(
          ValidationExample(
              window_id=int(window_id),
              label_type=annotation.label_type if annotation else None,
              score=float(logit),
              bin=int(bin_number),
              bin_weight=float(bin_weights[bin_number]),
          )
      )

  def convert_bins_to_search_results(
      self,
      samples_per_bin: int | None = None,
      skip_annotated: bool = False,
  ) -> search_results.TopKSearchResults:
    """Convert bins to search results."""

    if samples_per_bin is None:
      samples_per_bin = self.samples_per_bin

    results = []
    for examples in self.bins_dict.values():
      for ex in examples[:samples_per_bin]:
        if skip_annotated and ex.label_type is not None:
          continue
        results.append(
            search_results.SearchResult(
                window_id=ex.window_id,
                sort_score=ex.score,
                display_label_type=ex.label_type,
            )
        )

    return search_results.TopKSearchResults(
        top_k=len(results), search_results=results
    )

  def select_validation_examples(
      self,
      label_types: Sequence[interface.LabelType | None] = (
          interface.LabelType.POSITIVE,
          interface.LabelType.NEGATIVE,
      ),
  ) -> list[ValidationExample]:
    """Select validation examples with the given label types.

    Args:
      label_types: Label types to select from the study. Label types can be one
        of the LabelType enum values (POSITIVE, NEGATIVE, UNCERTAIN), or None
        (in which case unlabeled examples are included as well). By default,
        only POSITIVE and NEGATIVE examples are selected.

    Returns:
      List of ValidationExample objects.
    """

    validation_examples = []
    for examples in self.bins_dict.values():
      for ex in examples:
        if ex.label_type in label_types:
          validation_examples.append(ex)
    return validation_examples

  def update_from_annotated_windows(
      self,
      annotations_dict: dict[int, list[interface.Annotation]],
  ) -> None:
    """Update examples from annotations."""

    for examples in self.bins_dict.values():
      for ex in examples:
        annotations = annotations_dict.get(ex.window_id)
        if annotations:
          ex.label_type = annotations[0].label_type


def estimate_call_density(
    examples: list[ValidationExample],
    num_beta_samples: int = 10_000,
    beta_prior: float = 0.1,
) -> tuple[float, np.ndarray]:
  """Estimates call density from a set of ValidationExample.

  Args:
    examples: Validated examples.
    num_beta_samples: Number of times to draw from beta distributions.
    beta_prior: Prior for beta distribution.

  Returns:
    Expected value of density and an array of all sampled density estimates.
  """

  # Collect validated labels by bin.
  bin_pos = collections.defaultdict(int)
  bin_neg = collections.defaultdict(int)
  bin_weights = collections.defaultdict(float)
  for ex in examples:
    bin_weights[ex.bin] = ex.bin_weight
    if ex.label_type == interface.LabelType.POSITIVE:
      bin_pos[ex.bin] += 1
    elif ex.label_type == interface.LabelType.NEGATIVE:
      bin_neg[ex.bin] += 1

  # Create beta distributions.
  betas = {}
  for b in bin_weights:
    betas[b] = scipy.stats.beta(
        bin_pos[b] + beta_prior, bin_neg[b] + beta_prior
    )

  # MLE positive rate in each bin.
  density_ev = np.array([
      bin_weights[b] * bin_pos[b] / (bin_pos[b] + bin_neg[b] + 1e-6)
      for b in bin_weights
  ]).sum()

  q_betas = []
  for _ in range(num_beta_samples):
    q_beta = np.array([
        bin_weights[b] * betas[b].rvs(size=1)[0] for b in betas  # p(b) * P(+|b)
    ]).sum()
    q_betas.append(q_beta)

  return density_ev, np.array(q_betas)


def estimate_roc_auc(
    examples: list[ValidationExample],
) -> float:
  """Estimate the classifier ROC-AUC from validation logs.

  We use the probabilistic interpretation of ROC-AUC, as the probability that
  a uniformly sampled positive example has higher score than a uniformly sampled
  negative example.

  Abusing notation a bit, we decompose this as follows:
  P(+ > -) = sum_{b,c} P(+ > - | + in b, - in c) * P(b|+) * P(c|-)
  using the law of total probability and independence of drawing the pos/neg
  examples uniformly at random.

  When comparing scores from different bins b > c, we have all scores from b
  greater than scores from c, so P(+ > - | + in b, - in c) = 1.
  Likewise, if b < c, P(+ > - | + in b, - in c) = 0.

  When b == c, we can count directly the number of + > - pairs to estimate the
  in-bin ROC-AUC.

  The scalars P(b|+) and P(c|-) are computed using Bayes' rule and the expected
  value estimate of P(+).

  Args:
    examples: List of ValidationExample objects.

  Returns:
    ROC-AUC estimate.
  """

  # Collect validated labels by bin.
  bin_pos = collections.defaultdict(int)
  bin_neg = collections.defaultdict(int)
  bin_weights = collections.defaultdict(float)
  for ex in examples:
    bin_weights[ex.bin] = ex.bin_weight
    if ex.label_type == interface.LabelType.POSITIVE:
      bin_pos[ex.bin] += 1
    elif ex.label_type == interface.LabelType.NEGATIVE:
      bin_neg[ex.bin] += 1

  # P(+|b), P(-|b)
  p_pos_bin = {
      b: bin_pos[b] / (bin_pos[b] + bin_neg[b] + 1e-6) for b in bin_weights
  }
  p_neg_bin = {
      b: bin_neg[b] / (bin_pos[b] + bin_neg[b] + 1e-6) for b in bin_weights
  }
  # P(+), P(-) expected value.
  density_ev = np.array(
      [bin_weights[b] * p_pos_bin[b] for b in bin_weights]
  ).sum()
  p_bin_pos = collections.defaultdict(float)
  p_bin_neg = collections.defaultdict(float)
  for b in bin_weights:
    p_bin_pos[b] = bin_weights[b] * p_pos_bin[b] / density_ev
    p_bin_neg[b] = bin_weights[b] * p_neg_bin[b] / (1.0 - density_ev)

  roc_auc = 0

  # For off-diagonal bin pairs:
  # Take the probability of drawing a pos from bin j and neg from bin i.
  # If j > i, all pos examples are scored higher, so contributes directly to the
  # total ROC-AUC.
  bins = sorted(tuple(bin_weights.keys()))
  for i in range(len(bins)):
    for j in range(i + 1, len(bins)):
      roc_auc += p_bin_pos[j] * p_bin_neg[i]

  # For diagonal bin-pairs:
  # Look at actual in-bin observations for diagonal contribution.
  for b in bins:
    bin_pos_scores = np.array([
        v.score
        for v in examples
        if v.bin == b and v.label_type == interface.LabelType.POSITIVE
    ])
    bin_neg_scores = np.array([
        v.score
        for v in examples
        if v.bin == b and v.label_type == interface.LabelType.NEGATIVE
    ])
    # If either is empty, there's no chance of pulling a (pos, neg) pair from
    # this bin, so we can continue.
    if bin_pos_scores.size == 0 or bin_neg_scores.size == 0:
      continue
    # Count the total number of (pos, neg) pairs where a pos example has a
    # higher score than a negative example.
    hits = (
        (bin_pos_scores[:, np.newaxis] - bin_neg_scores[np.newaxis, :]) > 0
    ).sum()
    bin_roc_auc = hits / (bin_pos_scores.size * bin_neg_scores.size)

    # Contribution is the probability of pulling both pos and neg examples
    # from this bin, multiplied by the bin's ROC-AUC.
    roc_auc += bin_roc_auc * p_bin_pos[b] * p_bin_neg[b]

  return roc_auc
