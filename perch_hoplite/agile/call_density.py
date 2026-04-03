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

import abc
import collections
import dataclasses

from ml_collections import config_dict
import numpy as np
from perch_hoplite.agile import classifier as classifier_lib
from perch_hoplite.agile import embed
from perch_hoplite.db import datatypes
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
    sample_weight: Weight of the validation example.
    bin: Bin number the validation example was assigned to (ATB only).
  """

  window_id: int
  label_type: datatypes.LabelType | None
  score: float
  sample_weight: float
  bin: int | None = None


@dataclasses.dataclass
class ValidationSet:
  """A set of validation examples."""

  examples: list[ValidationExample]

  def append(self, example: ValidationExample):
    """Append a validation example to the set."""
    self.examples.append(example)

  def __iter__(self):
    """Iterate over the validation examples."""
    return iter(self.examples)

  @classmethod
  def from_config_dict(cls, config):
    if isinstance(config, config_dict.ConfigDict):
      config = config.to_dict()
    return cls(examples=[ValidationExample(**e) for e in config['examples']])

  def to_search_results(
      self,
      skip_annotated: bool = False,
  ) -> search_results.TopKSearchResults:
    """Convert bins to search results."""
    results = []
    for ex in self.examples:
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

  def update_annotations_from_db(
      self,
      db: interface.HopliteDBInterface,
      target_class: str,
  ) -> list[datatypes.Annotation | None]:
    """Returns annotations for the validation examples."""
    # Get existing annotations for validation examples from the database.
    # If more than one annotation is present, we pick the last one.
    annotations = []
    for ex in self.examples:
      window = db.get_window(ex.window_id)
      matching_annotations = db.get_all_annotations(
          filter=config_dict.create(
              eq=dict(
                  recording_id=window.recording_id,
                  label=target_class,
              ),
              approx=dict(offsets=window.offsets),
          )
      )
      if matching_annotations:
        ex.label_type = matching_annotations[-1].label_type
      annotations.append(ex.label_type)
    return annotations


class CallDensityEstimator(abc.ABC):
  """Abstract base class for call density estimation methods.

  A call density estimation workflow consists of N+1 steps:
  0. Specify parameters for the estimation task.
  1. Selecting validation examples for density estimation.
  2. Obtain labels for the validation examples though annotation.
  3. Estimate call density for some specified subset of data using the labeled
     validation examples.
  """

  @property
  @abc.abstractmethod
  def get_study_name(self) -> str:
    """Returns the name of the study."""

  @property
  @abc.abstractmethod
  def get_target_class(self) -> str:
    """Returns the target class."""

  @property
  @abc.abstractmethod
  def cde_method(self) -> str:
    """Returns the name of the call density estimation method."""

  @abc.abstractmethod
  def get_validation_examples(self) -> ValidationSet:
    """Collect selected validation examples."""

  @abc.abstractmethod
  def estimate_call_density(
      self,
      db: interface.HopliteDBInterface,
      **kwargs,
  ) -> tuple[float, np.ndarray]:
    """Estimates call density from a list of ValidationExample."""

  @abc.abstractmethod
  def estimate_roc_auc(self) -> float:
    """Estimate the classifier ROC-AUC from validation logs."""

  @abc.abstractmethod
  def to_config_dict(self) -> config_dict.ConfigDict:
    """Returns a ConfigDict representation of the study."""

  @classmethod
  @abc.abstractmethod
  def from_config_dict(
      cls,
      db: interface.HopliteDBInterface,
      config: config_dict.ConfigDict,
  ) -> 'CallDensityEstimator':
    """Instantiates a CallDensityEstimator from a ConfigDict."""

  def save_to_db(self, db: interface.HopliteDBInterface):
    """Saves the CallDensityEstimator to the database."""
    try:
      call_density_configs = db.get_metadata('call_density_configs')
    except KeyError:
      call_density_configs = config_dict.create()
    study_db_key = (
        f'{self.get_study_name}/{self.get_target_class}/{self.cde_method}'
    )
    call_density_configs[study_db_key] = self.to_config_dict()
    db.insert_metadata('call_density_configs', call_density_configs)
    db.commit()

  @classmethod
  def load_from_db(
      cls, db: interface.HopliteDBInterface, study_name: str, target_class: str
  ) -> 'CallDensityEstimator':
    """Loads the CallDensityEstimator from the database."""
    study_db_key = f'{study_name}/{target_class}/{cls.cde_method}'
    cfg = db.get_metadata('call_density_configs')[study_db_key]
    return cls.from_config_dict(db, cfg)


@dataclasses.dataclass(kw_only=True)
class CallDensityATB(datatypes.HopliteConfig, CallDensityEstimator):
  """Call density estimation using stratified sampling.

  Implements call density estimation using logarithmic binning and beta
  distributions, as described in `All Thresholds Barred', Navine, et al. (2024).
  """

  study_name: str
  classifier: classifier_lib.LinearClassifier
  target_class: str

  # ATB specific parameters.
  quantile_bounds: list[float]
  value_bounds: list[float]
  samples_per_bin: int
  validation_examples: ValidationSet

  # Filters used for selecting windows for density estimation.
  deployments_filter: config_dict.ConfigDict | None = None
  recordings_filter: config_dict.ConfigDict | None = None
  windows_filter: config_dict.ConfigDict | None = None

  @property
  def cde_method(self) -> str:
    """Returns the name of the call density estimation method."""
    return 'atb'

  @classmethod
  def load_from_db(
      cls, db: interface.HopliteDBInterface, study_name: str, target_class: str
  ) -> 'CallDensityEstimator':
    """Loads the CallDensityEstimator from the database."""
    study_db_key = f'{study_name}/{target_class}/atb'
    cfg = db.get_metadata('call_density_configs')[study_db_key]
    return cls.from_config_dict(db, cfg)

  @classmethod
  def create(
      cls,
      db: interface.HopliteDBInterface,
      rng_seed: int,
      study_name: str,
      classifier: classifier_lib.LinearClassifier,
      target_class: str,
      bin_bounds: list[float],
      binning_strategy: str,
      samples_per_bin: int,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      binning_sample_size: int = 0,
  ):
    """Creates a CallDensityATB object.

    Args:
      db: Hoplite database interface.
      rng_seed: Random seed for shuffling.
      study_name: Name of the study.
      classifier: Classifier to use for density estimation.
      target_class: Target class for density estimation.
      bin_bounds: Bounds for binning.
      binning_strategy: How to interpret the bin bounds, either 'quantile' or
        'value'.
      samples_per_bin: Number of samples per bin.
      deployments_filter: Filter for deployments.
      recordings_filter: Filter for recordings.
      windows_filter: Filter for windows.
      binning_sample_size: Number of samples to use for binning.
    """
    window_ids = db.match_window_ids(
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        windows_filter=windows_filter,
    )
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(window_ids)
    if binning_sample_size > 0:
      window_ids = window_ids[:binning_sample_size]

    # Compute logits for matching windows.
    target_class_idx = classifier.classes.index(target_class)
    logits = []
    for window_ids_batch in embed.batched(window_ids, 256):
      embeddings_batch = db.get_embeddings_batch(window_ids_batch)
      logits_batch = classifier(embeddings_batch)
      logits_batch = logits_batch[..., target_class_idx]
      logits.extend(logits_batch)
    logits = np.array(logits)

    if binning_strategy == 'quantile':
      quantile_bounds = bin_bounds
      value_bounds = np.quantile(logits, quantile_bounds).tolist()
    elif binning_strategy == 'value':
      value_bounds = bin_bounds
      quantile_bounds = []
      for value_bound in value_bounds:
        quantile_bounds.extend((logits < value_bound).sum() / len(logits))
      quantile_bounds.append(1.0)
    else:
      raise ValueError(f'Unknown binning strategy: {binning_strategy}')

    bin_numbers = np.digitize(logits, value_bounds)
    bin_weights = np.diff(quantile_bounds, prepend=-1.0)

    bins_dict = {bin: [] for bin in range(1, len(value_bounds))}
    for bin_number, window_id, logit in zip(bin_numbers, window_ids, logits):
      if bin_number == 0:
        bin_number = 1
      elif bin_number == len(value_bounds):
        bin_number -= 1
      bins_dict[bin_number].append(
          ValidationExample(
              window_id=int(window_id),
              label_type=None,
              score=float(logit),
              bin=int(bin_number),
              sample_weight=float(bin_weights[bin_number]),
          )
      )

    # Shuffle each bin and truncate to samples_per_bin.
    for bin_number in bins_dict:
      rng.shuffle(bins_dict[bin_number])
      bins_dict[bin_number] = bins_dict[bin_number][:samples_per_bin]
    examples = []
    for bin_number in bins_dict:
      examples.extend(bins_dict[bin_number])
    examples = ValidationSet(examples=examples)

    # Add any annotations already in the DB.
    examples.update_annotations_from_db(db, target_class)

    return cls(
        study_name=study_name,
        classifier=classifier,
        target_class=target_class,
        quantile_bounds=quantile_bounds,
        value_bounds=value_bounds,
        samples_per_bin=samples_per_bin,
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        windows_filter=windows_filter,
        validation_examples=examples,
    )

  @property
  def get_study_name(self) -> str:
    return self.study_name

  @property
  def get_target_class(self) -> str:
    return self.target_class

  @classmethod
  def from_config_dict(
      cls, db: interface.HopliteDBInterface, config: config_dict.ConfigDict
  ) -> 'CallDensityEstimator':
    """Creates a CallDensityEstimator from a ConfigDict."""
    kwargs = config.to_dict()
    kwargs['classifier']['beta'] = np.array(
        kwargs['classifier']['beta'], dtype=float
    )
    kwargs['classifier']['beta_bias'] = np.array(
        kwargs['classifier']['beta_bias'], dtype=float
    )
    kwargs['classifier'] = classifier_lib.LinearClassifier(
        **kwargs['classifier']
    )
    kwargs['validation_examples'] = ValidationSet.from_config_dict(
        kwargs['validation_examples']
    )
    return cls(**kwargs)

  def to_config_dict(self) -> config_dict.ConfigDict:
    """Returns a ConfigDict representation of the method."""
    d = dataclasses.asdict(self)
    d['classifier']['beta'] = d['classifier']['beta'].tolist()
    d['classifier']['beta_bias'] = d['classifier']['beta_bias'].tolist()
    return config_dict.ConfigDict(d)

  def get_validation_examples(
      self,
  ) -> ValidationSet:
    """Selects validation examples from bins_dict for density estimation."""
    return self.validation_examples

  def estimate_call_density(
      self,
      db: interface.HopliteDBInterface,
      num_beta_samples: int = 10_000,
      beta_prior: float = 0.1,
      default_label_type: datatypes.LabelType = datatypes.LabelType.UNCERTAIN,
      verbose=True,
  ) -> tuple[float, np.ndarray]:
    """Estimates call density from a list of ValidationExample.

    Args:
      db: Hoplite database interface.
      num_beta_samples: Number of samples to draw from beta distributions.
      beta_prior: Prior count for beta distributions.
      default_label_type: Default label type for unannotated windows.
      verbose: Whether to print verbose output.

    Returns:
      EV call density, and array of call density samples.
    """
    # Collect validated labels by bin.
    bin_pos = collections.defaultdict(int)
    bin_neg = collections.defaultdict(int)
    bin_weights = collections.defaultdict(float)
    self.validation_examples.update_annotations_from_db(db, self.target_class)
    for ex in self.validation_examples:
      bin_weights[ex.bin] = ex.sample_weight
      if ex.label_type is None:
        ex.label_type = default_label_type
      if ex.label_type == datatypes.LabelType.POSITIVE:
        bin_pos[ex.bin] += 1
      elif ex.label_type == datatypes.LabelType.NEGATIVE:
        bin_neg[ex.bin] += 1

    if verbose:
      print('Bin positive counts : ', bin_pos)
      print('Bin negative counts : ', bin_neg)
      print('Bin weights         : ', bin_weights)

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
          bin_weights[b] * betas[b].rvs(size=1)[0]
          for b in betas  # p(b) * P(+|b)
      ]).sum()
      q_betas.append(q_beta)

    return density_ev, np.array(q_betas)

  def estimate_roc_auc(self) -> float:
    """Estimate the classifier ROC-AUC from validation logs.

    We use the probabilistic interpretation of ROC-AUC, as the probability that
    a uniformly sampled positive example has higher score than a uniformly
    sampled negative example.

    Abusing notation a bit, we decompose this as follows:
    P(+ > -) = sum_{b,c} P(+ > - | + in b, - in c) * P(b|+) * P(c|-)
    using the law of total probability and independence of drawing the pos/neg
    examples uniformly at random.

    When comparing scores from different bins b > c, we have all scores from b
    greater than scores from c, so P(+ > - | + in b, - in c) = 1.
    Likewise, if b < c, P(+ > - | + in b, - in c) = 0.

    When b == c, we can count directly the number of + > - pairs to estimate the
    in-bin ROC-AUC.

    The scalars P(b|+) and P(c|-) are computed using Bayes' rule and the
    expected
    value estimate of P(+).

    Returns:
      ROC-AUC estimate.
    """

    # Collect validated labels by bin.
    bin_pos = collections.defaultdict(int)
    bin_neg = collections.defaultdict(int)
    bin_weights = collections.defaultdict(float)
    for ex in self.validation_examples:
      bin_weights[ex.bin] = ex.sample_weight
      if ex.label_type == datatypes.LabelType.POSITIVE:
        bin_pos[ex.bin] += 1
      elif ex.label_type == datatypes.LabelType.NEGATIVE:
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
    # If j > i, all pos examples are scored higher, so contributes directly
    # to the total ROC-AUC.
    bins = sorted(tuple(bin_weights.keys()))
    for i in range(len(bins)):
      for j in range(i + 1, len(bins)):
        roc_auc += p_bin_pos[j] * p_bin_neg[i]

    # For diagonal bin-pairs:
    # Look at actual in-bin observations for diagonal contribution.
    for b in bins:
      bin_pos_scores = np.array([
          v.score
          for v in self.validation_examples
          if v.bin == b and v.label_type == datatypes.LabelType.POSITIVE
      ])
      bin_neg_scores = np.array([
          v.score
          for v in self.validation_examples
          if v.bin == b and v.label_type == datatypes.LabelType.NEGATIVE
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
