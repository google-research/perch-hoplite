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

"""Metrics for training and validation."""

from typing import Any, Dict

import numpy as np


def map_(
    logits: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray | None = None,
    sort_descending: bool = True,
) -> np.ndarray:
  return average_precision(
      scores=logits,
      labels=labels,
      label_mask=label_mask,
      sort_descending=sort_descending,
  )


def cmap(
    logits: np.ndarray,
    labels: np.ndarray,
    sort_descending: bool = True,
    sample_threshold: int = 0,
) -> Dict[str, Any]:
  """Class mean average precision."""
  class_aps = average_precision(
      scores=logits.T, labels=labels.T, sort_descending=sort_descending
  )
  mask = np.sum(labels, axis=0) > sample_threshold
  class_aps = np.where(mask, class_aps, np.nan)
  macro_cmap = np.mean(class_aps, where=mask)
  return {
      'macro': macro_cmap,
      'individual': class_aps,
  }


def roc_auc(
    logits: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray | None = None,
    sort_descending: bool = True,
    sample_threshold: int = 1,
) -> Dict[str, Any]:
  """Computes ROC-AUC scores.

  Args:
    logits: A score for each label which can be ranked.
    labels: A multi-hot encoding of the ground truth positives. Must match the
      shape of scores.
    label_mask: A mask indicating which labels to involve in the calculation.
    sort_descending: An indicator if the search result ordering is in descending
      order (e.g. for evaluating over similarity metrics where higher scores are
      preferred). If false, computes average_precision on descendingly sorted
      inputs.
    sample_threshold: Only classes with at least this many samples will be used
      in the calculation of the final metric. By default this is 1, which means
      that classes without any positive examples will be ignored.

  Returns:
    A dictionary of ROC-AUC scores using the arithmetic ('macro') and
    geometric means, along with individual class ('individual') ROC-AUC and its
    variance.
  """
  if label_mask is not None:
    label_mask = label_mask.T
  class_roc_auc, class_roc_auc_var = generalized_mean_rank(
      logits.T, labels.T, label_mask=label_mask, sort_descending=sort_descending
  )
  mask = np.sum(labels, axis=0) >= sample_threshold
  class_roc_auc = np.where(mask, class_roc_auc, np.nan)
  class_roc_auc_var = np.where(mask, class_roc_auc_var, np.nan)
  return {
      'macro': np.mean(class_roc_auc, where=mask),
      'geometric': np.exp(np.mean(np.log(class_roc_auc), where=mask)),
      'individual': class_roc_auc,
      'individual_var': class_roc_auc_var,
  }


def average_precision(
    scores: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray | None = None,
    sort_descending: bool = True,
    interpolated: bool = False,
) -> np.ndarray:
  """Average precision.

  The average precision is the area under the precision-recall curve. When
  using interpolation we take the maximum precision over all smaller recalls.
  The intuition is that it often makes sense to evaluate more documents if the
  total percentage of relevant documents increases.
  Average precision is computed over the last axis.

  Args:
    scores: A score for each label which can be ranked.
    labels: A multi-hot encoding of the ground truth positives. Must match the
      shape of scores.
    label_mask: A mask indicating which labels to involve in the calculation.
    sort_descending: An indicator if the search result ordering is in descending
      order (e.g. for evaluating over similarity metrics where higher scores are
      preferred). If false, computes average_precision on descendingly sorted
      inputs.
    interpolated: Whether to use interpolation.

  Returns:
    The average precision.
  """
  if label_mask is not None:
    # Set all masked labels to zero, and send the scores for those labels to a
    # low/high value (depending on whether we sort in descending order or not).
    # Then the masked scores+labels will not impact the average precision
    # calculation.
    labels = labels * label_mask
    extremum_score = (
        np.min(scores) - 1.0 if sort_descending else np.max(scores) + 1.0
    )
    scores = np.where(label_mask, scores, extremum_score)
  idx = np.argsort(scores)
  if sort_descending:
    idx = np.flip(idx, axis=-1)
  scores = np.take_along_axis(scores, idx, axis=-1)
  labels = np.take_along_axis(labels, idx, axis=-1)
  pr_curve = np.cumsum(labels, axis=-1) / (np.arange(labels.shape[-1]) + 1)
  if interpolated:
    pr_curve = -np.maximum.accumulate(-pr_curve, axis=-1)

  # In case of an empty row, assign precision = 0, and avoid dividing by zero.
  mask = np.float32(np.sum(labels, axis=-1) != 0)
  raw_av_prec = np.sum(pr_curve * labels, axis=-1) / np.maximum(
      np.sum(labels, axis=-1), 1.0
  )
  return mask * raw_av_prec


def generalized_mean_rank(
    scores: np.ndarray,
    labels: np.ndarray,
    label_mask: np.ndarray | None = None,
    sort_descending: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes the generalized mean rank and its variance over the last axis.

  The generalized mean rank can be expressed as

      (sum_i #P ranked above N_i) / (#P * #N),

  or equivalently,

      1 - (sum_i #N ranked above P_i) / (#P * #N).

  This metric is usually better visualized in the logits domain, where it
  reflects the log-odds of ranking a randomly-chosen positive higher than a
  randomly-chosen negative.

  Args:
    scores: A score for each label which can be ranked.
    labels: A multi-hot encoding of the ground truth positives. Must match the
      shape of scores.
    label_mask: A mask indicating which labels to involve in the calculation.
    sort_descending: An indicator if the search result ordering is in descending
      order (e.g. for evaluating over similarity metrics where higher scores are
      preferred). If false, computes the generalize mean rank on descendingly
      sorted inputs.

  Returns:
    The generalized mean rank and its variance. The variance is calculated by
    considering each positive to be an independent sample of the value
    1 - #N ranked above P_i / #N. This gives a measure of how consistently
    positives are ranked.
  """
  idx = np.argsort(scores, axis=-1)
  if sort_descending:
    idx = np.flip(idx, axis=-1)
  labels = np.take_along_axis(labels, idx, axis=-1)
  if label_mask is None:
    label_mask = True
  else:
    label_mask = np.take_along_axis(label_mask, idx, axis=-1)

  num_p = (labels > 0).sum(axis=-1, where=label_mask)
  num_p_above = np.cumsum((labels > 0) & label_mask, axis=-1)
  num_n = (labels == 0).sum(axis=-1, where=label_mask)
  num_n_above = np.cumsum((labels == 0) & label_mask, axis=-1)

  gmr = num_p_above.mean(axis=-1, where=(labels == 0) & label_mask) / num_p
  gmr_var = (num_n_above / num_n[:, None]).var(
      axis=-1, where=(labels > 0) & label_mask
  )
  return gmr, gmr_var
