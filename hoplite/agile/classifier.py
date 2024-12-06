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

"""Functions for training and applying a linear classifier."""

from typing import Any

from hoplite.agile import classifier_data
from hoplite.agile import metrics
import numpy as np
import tensorflow as tf
import tqdm


def get_linear_model(embedding_dim: int, num_classes: int) -> tf.keras.Model:
  """Create a simple linear Keras model."""
  model = tf.keras.Sequential([
      tf.keras.Input(shape=[embedding_dim]),
      tf.keras.layers.Dense(num_classes),
  ])
  return model


def bce_loss(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    is_labeled_mask: tf.Tensor,
    weak_neg_weight: float,
) -> tf.Tensor:
  """Binary cross entropy loss from logits with weak negative weights."""
  y_true = tf.cast(y_true, dtype=logits.dtype)
  log_p = tf.math.log_sigmoid(logits)
  log_not_p = tf.math.log_sigmoid(-logits)
  # optax sigmoid_binary_cross_entropy:
  # -labels * log_p - (1.0 - labels) * log_not_p
  raw_bce = -y_true * log_p - (1.0 - y_true) * log_not_p
  is_labeled_mask = tf.cast(is_labeled_mask, dtype=logits.dtype)
  weights = (1.0 - is_labeled_mask) * weak_neg_weight + is_labeled_mask
  return tf.reduce_mean(raw_bce * weights)


def hinge_loss(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    is_labeled_mask: tf.Tensor,
    weak_neg_weight: float,
) -> tf.Tensor:
  """Weighted SVM hinge loss."""
  # Convert multihot to +/- 1 labels.
  y_true = 2 * y_true - 1
  weights = (1.0 - is_labeled_mask) * weak_neg_weight + is_labeled_mask
  raw_hinge_loss = tf.maximum(0, 1 - y_true * logits)
  return tf.reduce_mean(raw_hinge_loss * weights)


def infer(params, embeddings: np.ndarray):
  """Apply the model to embeddings."""
  return np.dot(embeddings, params['beta']) + params['beta_bias']


def eval_classifier(
    params: Any,
    data_manager: classifier_data.DataManager,
    eval_ids: np.ndarray,
) -> dict[str, float]:
  """Evaluate a classifier on a set of examples."""
  iter_ = data_manager.batched_example_iterator(
      eval_ids, add_weak_negatives=False, repeat=False
  )
  # The embedding ids may be shuffled by the iterator, so we will track the ids
  # of the examples we are evaluating.
  got_ids = []
  pred_logits = []
  true_labels = []
  for batch in iter_:
    pred_logits.append(infer(params, batch.embedding))
    true_labels.append(batch.multihot)
    got_ids.append(batch.idx)
  pred_logits = np.concatenate(pred_logits, axis=0)
  true_labels = np.concatenate(true_labels, axis=0)
  got_ids = np.concatenate(got_ids, axis=0)

  # Compute the top1 accuracy on examples with at least one label.
  labeled_locs = np.where(true_labels.sum(axis=1) > 0)
  top_preds = np.argmax(pred_logits, axis=1)
  top1 = true_labels[np.arange(top_preds.shape[0]), top_preds]
  top1 = top1[labeled_locs].mean()

  rocs = metrics.roc_auc(
      logits=pred_logits, labels=true_labels, sample_threshold=1
  )
  cmaps = metrics.cmap(
      logits=pred_logits, labels=true_labels, sample_threshold=1
  )
  return {
      'top1_acc': top1,
      'roc_auc': rocs['macro'],
      'roc_auc_individual': rocs['individual'],
      'cmap': cmaps['macro'],
      'cmap_individual': cmaps['individual'],
      'eval_ids': got_ids,
      'eval_preds': pred_logits,
      'eval_labels': true_labels,
  }


def train_linear_classifier(
    data_manager: classifier_data.DataManager,
    learning_rate: float,
    weak_neg_weight: float,
    num_train_steps: int,
    loss: str = 'bce',
):
  """Train a linear classifier."""
  embedding_dim = data_manager.db.embedding_dimension()
  num_classes = len(data_manager.get_target_labels())
  lin_model = get_linear_model(embedding_dim, num_classes)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  lin_model.compile(optimizer=optimizer, loss='binary_crossentropy')
  if loss == 'hinge':
    loss_fn = hinge_loss
  elif loss == 'bce':
    loss_fn = bce_loss
  else:
    raise ValueError(f'Unknown loss: {loss}')

  @tf.function
  def train_step(y_true, embeddings, is_labeled_mask):
    with tf.GradientTape() as tape:
      logits = lin_model(embeddings, training=True)
      loss = loss_fn(y_true, logits, is_labeled_mask, weak_neg_weight)
      loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, lin_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, lin_model.trainable_variables))
    return loss

  train_idxes, eval_idxes = data_manager.get_train_test_split()
  train_iter_ = data_manager.batched_example_iterator(
      train_idxes, add_weak_negatives=True, repeat=True
  )
  progress = tqdm.tqdm(enumerate(train_iter_), total=num_train_steps)
  update_steps = set([b * (num_train_steps // 100) for b in range(100)])

  for step, batch in enumerate(train_iter_):
    if step >= num_train_steps:
      break
    step_loss = train_step(
        batch.multihot, batch.embedding, batch.is_labeled_mask
    )
    if step in update_steps:
      progress.update(n=num_train_steps // 100)
      progress.set_description(f'Loss {step_loss:.8f}')
  progress.clear()
  progress.close()

  params = {
      'beta': lin_model.get_weights()[0],
      'beta_bias': lin_model.get_weights()[1],
  }
  eval_scores = eval_classifier(params, data_manager, eval_idxes)
  return params, eval_scores
