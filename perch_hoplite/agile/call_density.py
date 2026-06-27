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
from scipy import stats
from sklearn import linear_model


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
      matching_annotations = db.get_window_annotations(
          window.id, label=target_class
      )
      if matching_annotations:
        ex.label_type = matching_annotations[-1].label_type
      annotations.append(ex.label_type)
    return annotations

  def to_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns np.arrays of scores, labels, and weights."""
    scores = []
    labels = []
    weights = []
    for ex in self.examples:
      if ex.label_type is None:
        continue
      if ex.label_type in [
          datatypes.LabelType.POSITIVE,
          datatypes.LabelType.NEGATIVE,
      ]:
        scores.append(ex.score)
        labels.append(ex.label_type.value)  # pytype: disable=attribute-error
        weights.append(ex.sample_weight)
    return np.array(scores), np.array(labels), np.array(weights)

  def bootstrap_sample(
      self,
      rng: np.random.Generator,
  ) -> 'ValidationSet':
    """Returns a list of bootstrap samples."""
    indices = rng.choice(len(self.examples), size=len(self.examples))
    return ValidationSet(examples=[self.examples[i] for i in indices])


@dataclasses.dataclass(kw_only=True)
class CallDensityEstimator(datatypes.HopliteConfig, abc.ABC):
  """Abstract base class for call density estimation methods.

  A call density estimation workflow consists of N+1 steps:
  0. Specify parameters for the estimation task.
  1. Selecting validation examples for density estimation.
  2. Obtain labels for the validation examples though annotation.
  3. Estimate call density for some specified subset of data using the labeled
     validation examples.
  """
  # TODO(tomdenton): Add a method for extending the validation set.
  study_name: str
  classifier: classifier_lib.LinearClassifier
  target_class: str

  validation_examples: ValidationSet

  @classmethod
  @abc.abstractmethod
  def cde_method(cls) -> str:
    """Returns the name of the call density estimation method."""

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

  def estimate_deployment_call_density(
      self,
      db: interface.HopliteDBInterface,
      **kwargs,
  ) -> dict[str, tuple[float, np.ndarray]]:
    """Estimates call density per deployment."""
    # TODO(tomdenton): Implement per-deployment call density estimation.
    raise NotImplementedError

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

  def save_to_db(self, db: interface.HopliteDBInterface):
    """Saves the CallDensityEstimator to the database."""
    try:
      call_density_configs = db.get_metadata('call_density_configs')
    except KeyError:
      call_density_configs = config_dict.create()
    study_db_key = f'{self.study_name}/{self.target_class}/{self.cde_method()}'
    call_density_configs[study_db_key] = self.to_config_dict()
    db.insert_metadata('call_density_configs', call_density_configs)
    db.commit()

  @classmethod
  def load_from_db(
      cls, db: interface.HopliteDBInterface, study_name: str, target_class: str
  ) -> 'CallDensityEstimator':
    """Loads the CallDensityEstimator from the database."""
    study_db_key = f'{study_name}/{target_class}/{cls.cde_method()}'
    cfg = db.get_metadata('call_density_configs')[study_db_key]
    return cls.from_config_dict(db, cfg)


@dataclasses.dataclass(kw_only=True)
class CallDensityPlattScaling(CallDensityEstimator):
  """Call density estimation using Hierarchical Platt scaling."""

  # Platt Scaling specific parameters.
  sampling_exponent: float
  sampling_epsilon: float

  # Filters used for selecting windows for density estimation.
  deployments_filter: config_dict.ConfigDict | None = None
  recordings_filter: config_dict.ConfigDict | None = None

  @classmethod
  def cde_method(cls) -> str:
    """Returns the name of the call density estimation method."""
    return 'platt_scaling'

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
      unlabeled_sample_size: int | None = None,
  ):
    """Creates a CallDensityHierarchicalPlatt object."""
    window_ids, logits = match_windows_and_compute_logits(
        db,
        rng_seed,
        classifier,
        target_class,
        deployments_filter,
        recordings_filter,
        sample_size=unlabeled_sample_size,
    )
    valid_window_ids, valid_logits, sample_weights = monomial_sample_selection(
        window_ids,
        logits,
        n_validation_samples,
        sampling_exponent,
        sampling_epsilon,
        deterministic=deterministic,
        rng_seed=rng_seed,
    )
    examples = []
    for window_id, logit, sample_weight in zip(
        valid_window_ids, valid_logits, sample_weights
    ):
      examples.append(
          ValidationExample(
              window_id=int(window_id),
              label_type=None,
              score=float(logit),
              sample_weight=float(sample_weight),
          )
      )
    examples = ValidationSet(examples=examples)
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

  def get_validation_examples(self) -> ValidationSet:
    """Collect selected validation examples."""
    return self.validation_examples

  def estimate_call_density(
      self,
      db: interface.HopliteDBInterface,
      rng_seed: int = 42,
      scores_sample_size: int | None = None,
      n_points: int = 1024,
      n_resamples: int = 1024,
      default_label_type: datatypes.LabelType = datatypes.LabelType.UNCERTAIN,
      l2_reg: float | None = 1.0,
  ) -> tuple[float, np.ndarray]:
    """Estimates call density from a list of ValidationExample."""
    rng = np.random.default_rng(rng_seed)
    self.validation_examples.update_annotations_from_db(db, self.target_class)
    for ex in self.validation_examples:
      if ex.label_type is None:
        ex.label_type = default_label_type
    scores, labels, weights = self.validation_examples.to_vectors()

    # Get a distribution of scores.
    _, unlabeled_scores = match_windows_and_compute_logits(
        db,
        rng_seed,
        self.classifier,
        self.target_class,
        self.deployments_filter,
        self.recordings_filter,
        sample_size=scores_sample_size,
    )
    xs, unlabeled_kde = fit_kde(unlabeled_scores, n_points)
    _, platt_pred = fit_logistic(
        scores, labels, xs, sample_weight=weights, l2_reg=l2_reg
    )
    p_pos_mle = np.sum(unlabeled_kde * platt_pred) * (xs[1] - xs[0])

    sampled_p_pos = []
    for _ in range(n_resamples):
      bootstrap_val = self.validation_examples.bootstrap_sample(rng)
      boot_scores, boot_labels, boot_weights = bootstrap_val.to_vectors()
      _, boot_platt_pred = fit_logistic(
          boot_scores,
          boot_labels,
          xs,
          sample_weight=boot_weights,
          l2_reg=l2_reg,
      )
      sampled_p_pos.append(
          np.sum(unlabeled_kde * boot_platt_pred) * (xs[1] - xs[0])
      )
    return p_pos_mle, np.array(sampled_p_pos)

  def estimate_deployment_call_density(
      self,
      db: interface.HopliteDBInterface,
      rng_seed: int = 42,
      n_resamples: int = 1024,
      default_label_type: datatypes.LabelType = datatypes.LabelType.UNCERTAIN,
      l2_reg: float | None = 1.0,
      alpha: float = 0.2,
      xs: np.ndarray | None = None,
      bootstrap_recordings: bool = True,
      model_activity: bool = True,
  ) -> dict[str, tuple[float, np.ndarray]]:
    """Estimates call density per deployment.

    Accounts for time-dependence within recordings by assigning an occupancy
    probability to each recording based on its maximum logit.

    Args:
      db: Hoplite database interface.
      rng_seed: Random seed for bootstrapping.
      n_resamples: Number of bootstrap resamples.
      default_label_type: Default label for unannotated validation examples.
      l2_reg: L2 regularization strength for Platt scaling.
      alpha: Mixing parameter for independent and max-logit occupancy models.
      xs: Evaluation points for Platt scaling.
      bootstrap_recordings: Whether to apply hierarchical bootstrapping to
        recordings, or treat windows within recordings as independent.
      model_activity: Whether to use hierarchical modeling of activity at the
        recording level.

    Returns:
      A dictionary mapping deployment IDs to
      (MLE estimate, bootstrap samples).
    """
    rng = np.random.default_rng(rng_seed)
    if xs is None:
      _, unlabeled_scores = match_windows_and_compute_logits(
          db,
          rng_seed,
          self.classifier,
          self.target_class,
          self.deployments_filter,
          self.recordings_filter,
          sample_size=2048,
      )
      xs = np.linspace(np.min(unlabeled_scores), np.max(unlabeled_scores), 1024)

    self.validation_examples.update_annotations_from_db(db, self.target_class)
    for ex in self.validation_examples:
      if ex.label_type is None:
        ex.label_type = default_label_type

    # 1. Fit Platt scaling models.
    def get_lr_params(lr_obj, scores, labels, weights, current_rng):
      if lr_obj is None:
        unique_labels = np.unique(labels)
        if len(unique_labels) == 1:
          n_pos = weights.sum() if unique_labels[0] == 1 else 0
          n_neg = weights.sum() if unique_labels[0] == 0 else 0
          return float(
              scipy.stats.beta(n_pos + 0.5, n_neg + 0.5).rvs(
                  random_state=current_rng
              )
          )
        return 0.5
      try:
        return lr_obj.coef_[0][0], lr_obj.intercept_[0]
      except (ValueError, AttributeError):
        return 0.5

    bs_params_list = []
    for _ in range(n_resamples):
      bs_val = self.validation_examples.bootstrap_sample(rng)
      bs_s, bs_l, bs_w = bs_val.to_vectors()
      lr_bs, _ = fit_logistic(bs_s, bs_l, xs, sample_weight=bs_w, l2_reg=l2_reg)
      bs_params_list.append(get_lr_params(lr_bs, bs_s, bs_l, bs_w, rng))

    # 2. Gather data per deployment/recording.
    matching_deployments = db.get_all_deployments(
        filter=self.deployments_filter
    )
    matching_deployment_ids = {d.id for d in matching_deployments}

    matching_recordings = db.get_all_recordings(filter=self.recordings_filter)
    depl_recs = collections.defaultdict(list)
    for r in matching_recordings:
      if r.deployment_id in matching_deployment_ids:
        depl_recs[r.deployment_id].append(r)

    matching_windows = db.get_all_windows(
        deployments_filter=self.deployments_filter,
        recordings_filter=self.recordings_filter,
    )
    rec_window_ids = collections.defaultdict(list)
    for w in matching_windows:
      rec_window_ids[w.recording_id].append(w.id)

    # 3. Compute logits.
    target_class_idx = self.classifier.classes.index(self.target_class)
    all_window_ids = [w.id for w in matching_windows]
    window_id_to_logit = {}
    for batch_ids in embed.batched(all_window_ids, 256):
      embs = db.get_embeddings_batch(batch_ids)
      logits_batch = self.classifier(embs)
      logits_batch = logits_batch[..., target_class_idx]
      for wid, l in zip(batch_ids, logits_batch):
        window_id_to_logit[wid] = l

    rec_logits = {}
    for rid, wids in rec_window_ids.items():
      rec_logits[rid] = np.array([window_id_to_logit[wid] for wid in wids])

    # 4. Helper for applying Platt scaling.
    def get_probs(logits, params):
      if isinstance(params, float):
        return np.full_like(logits, params)
      a, b = params
      return 1.0 / (1.0 + np.exp(-(a * logits + b)))

    # 5. Process deployments.
    results = {}
    for depl in matching_deployments:
      recs_in_depl = depl_recs[depl.id]
      # Filter for recordings that actually have windows.
      depl_logits = [
          rec_logits[r.id] for r in recs_in_depl if r.id in rec_logits
      ]
      if not depl_logits:
        results[depl.id] = (0.0, np.zeros(n_resamples))
        continue

      # Bootstrap resamples
      bs_densities = []
      for bs_params in bs_params_list:
        # Sample recordings with replacement.
        if bootstrap_recordings:
          indices = rng.choice(len(depl_logits), size=len(depl_logits))
        else:
          indices = range(len(depl_logits))
        total_bs_count = 0
        total_bs_windows = 0
        for idx in indices:
          r_logits = depl_logits[idx]
          probs = get_probs(r_logits, bs_params)
          if model_activity:
            p_occ_max = np.max(probs)
            p_occ_ind = 1.0 - np.prod(1.0 - probs)
            p_occ = alpha * p_occ_ind + (1.0 - alpha) * p_occ_max
          else:
            p_occ = 1.0
          # Stochastic generation for a single resample.
          total_bs_windows += len(r_logits)
          # total_bs_count += np.sum(probs) * (rng.random() < p_occ)
          total_bs_count += np.sum(rng.random(len(probs)) < probs) * (
              rng.random() < p_occ
          )
        bs_densities.append(total_bs_count / total_bs_windows)
      # Use median as the point estimate to better reflect obviously-empty
      # deployments.
      mle_density = np.median(bs_densities)

      results[depl.id] = (mle_density, np.array(bs_densities))

    return results

  def estimate_roc_auc(self) -> float:
    """Estimates ROC AUC from validation examples."""
    # TODO(tomdenton): Add support for weighted examples.
    scores, labels, unused_weights = self.validation_examples.to_vectors()
    pos_scores = scores[labels == datatypes.LabelType.POSITIVE.value]
    neg_scores = scores[labels == datatypes.LabelType.NEGATIVE.value]
    if pos_scores.size == 0 or neg_scores.size == 0:
      return 0.0
    hits = 0
    for ps in pos_scores:
      hits += (ps > neg_scores).sum()
      hits += (ps == neg_scores).sum() * 0.5
    roc_auc = hits / (pos_scores.size * neg_scores.size)
    return roc_auc


@dataclasses.dataclass(kw_only=True)
class CallDensityATB(CallDensityEstimator):
  """Call density estimation using stratified sampling.

  Implements call density estimation using logarithmic binning and beta
  distributions, as described in `All Thresholds Barred', Navine, et al. (2024).
  """

  # ATB specific parameters.
  quantile_bounds: list[float]
  value_bounds: list[float]
  samples_per_bin: int

  # Filters used for selecting windows for density estimation.
  deployments_filter: config_dict.ConfigDict | None = None
  recordings_filter: config_dict.ConfigDict | None = None

  @classmethod
  def cde_method(cls) -> str:
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
      binning_sample_size: int | None = None,
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
      binning_sample_size: Number of samples to use for binning.
    """
    window_ids, logits = match_windows_and_compute_logits(
        db,
        rng_seed,
        classifier,
        target_class,
        deployments_filter,
        recordings_filter,
        sample_size=binning_sample_size,
    )

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
    rng = np.random.default_rng(rng_seed + 1)
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
        validation_examples=examples,
    )

  def estimate_call_density(
      self,
      db: interface.HopliteDBInterface,
      n_resamples: int = 1024,
      beta_prior: float = 0.1,
      default_label_type: datatypes.LabelType = datatypes.LabelType.UNCERTAIN,
      verbose=False,
  ) -> tuple[float, np.ndarray]:
    """Estimates call density from a list of ValidationExample.

    Args:
      db: Hoplite database interface.
      n_resamples: Number of samples to draw from beta distributions.
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
    for _ in range(n_resamples):
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


def fit_logistic(
    scores: np.ndarray,
    labels: np.ndarray,
    xs: np.ndarray,
    sample_weight=None,
    l2_reg: float = 1.0,
) -> tuple[linear_model.LogisticRegression | None, np.ndarray]:
  """Fit a logistic regression model to the scores.

  Args:
    scores: Array of scores to fit the model to.
    labels: Array of labels to fit the model to.
    xs: Array of scores to predict the model on.
    sample_weight: Array of sample weights to use for fitting the model.
    l2_reg: L2 regularization strength.

  Returns:
    Logistic regression model, and array of predicted probabilities.
  """
  unique_labels = np.unique(labels)
  if len(unique_labels) == 1:
    if unique_labels[0] == 1:
      n_pos = sample_weight.sum() if sample_weight is not None else len(labels)
      n_neg = 0
    else:
      n_pos = 0
      n_neg = sample_weight.sum() if sample_weight is not None else len(labels)
    val = scipy.stats.beta(n_pos + 0.5, n_neg + 0.5).rvs()
    return None, np.full_like(xs, val)

  if len(scores.shape) == 1:
    scores = scores[:, np.newaxis]
  # Platt 1999 regularizes by modifying the targets.
  # sklearn uses L2 regulatrization, and doesn't support non-int targets.
  # n = labels.shape[0]
  # labels = (n * labels + 1) / (n + 2)
  lr = linear_model.LogisticRegression(max_iter=512, C=l2_reg)
  try:
    lr.fit(scores, labels, sample_weight=sample_weight)
    platt_pred = lr.predict_proba(xs[:, np.newaxis])[:, 1]
  except ValueError as e:
    print(f'Warning: Platt scaling failed: {e}')
    platt_pred = np.full_like(xs, 0.5)
  return lr, platt_pred


def fit_kde(scores, n_xs=1024, xs: np.ndarray | None = None):
  """Fit a KDE model to the scores.

  Args:
    scores: Array of scores to fit the model to.
    n_xs: Number of points to evaluate the KDE on.
    xs: Array of scores to evaluate the KDE on. If None, create a linspace from
      min(scores) to max(scores).

  Returns:
    Array of scores to evaluate the KDE on, and array of KDE values.
  """
  # Calculate the KDE
  kde = stats.gaussian_kde(scores)
  if xs is None:
    xs = np.linspace(min(scores), max(scores), n_xs)
  evaluated = kde.evaluate(xs)
  return xs, evaluated


def monomial_sample_selection(
    window_ids: np.ndarray,
    logits: np.ndarray,
    n_validation_samples: int,
    sampling_exponent: float,
    sampling_epsilon: float,
    deterministic: bool = True,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Selects samples using monomial sampling strategy."""
  s_idxes = np.argsort(logits)
  window_ids = window_ids[s_idxes]
  logits = logits[s_idxes]
  n_poly = int((1 - sampling_epsilon) * n_validation_samples)
  n_unif = n_validation_samples - n_poly
  if deterministic:
    qs_poly = np.linspace(1e-5, 1.0, num=n_poly, endpoint=False)
    qs_unif = np.linspace(1e-5, 1.0, num=n_unif, endpoint=False)
  else:
    rng = np.random.default_rng(rng_seed)
    qs_poly = rng.uniform(1e-5, 1.0, size=n_poly)
    qs_unif = rng.uniform(1e-5, 1.0, size=n_unif)
  qs = np.concatenate([qs_poly, qs_unif])
  inverse_zs = qs_poly ** (1.0 / sampling_exponent)
  inverse_zs = np.concatenate([inverse_zs, qs_unif])
  dzs = sampling_exponent * qs ** (1.0 - 1.0 / sampling_exponent)
  sample_weights = np.reciprocal(
      (1 - sampling_epsilon) * dzs + sampling_epsilon
  )
  z_args = np.int32(inverse_zs * logits.shape[0])
  valid_window_ids = window_ids[z_args]
  valid_logits = logits[z_args]
  return valid_window_ids, valid_logits, sample_weights


def match_windows_and_compute_logits(
    db: interface.HopliteDBInterface,
    rng_seed: int,
    classifier: classifier_lib.LinearClassifier,
    target_class: str,
    deployments_filter: config_dict.ConfigDict | None = None,
    recordings_filter: config_dict.ConfigDict | None = None,
    sample_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Matches windows from the database."""
  window_ids = db.match_window_ids(
      deployments_filter=deployments_filter,
      recordings_filter=recordings_filter,
  )
  rng = np.random.default_rng(rng_seed)
  window_ids = np.array(window_ids)
  rng.shuffle(window_ids)
  if sample_size is not None and sample_size > 0:
    window_ids = window_ids[:sample_size]

  # Compute logits for matching windows.
  target_class_idx = classifier.classes.index(target_class)
  logits = []
  for window_ids_batch in embed.batched(window_ids, 256):
    embeddings_batch = db.get_embeddings_batch(window_ids_batch)
    logits_batch = classifier(embeddings_batch)
    logits_batch = logits_batch[..., target_class_idx]
    logits.extend(logits_batch)
  logits = np.array(logits)
  return window_ids, logits
