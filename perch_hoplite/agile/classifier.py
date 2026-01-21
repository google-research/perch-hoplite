# coding=utf-8
# Copyright 2025 The Perch Authors.
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

import base64
from collections.abc import Iterator, Sequence
from concurrent import futures
import csv
import dataclasses
import json
from typing import Any

from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.agile import classifier_data
from perch_hoplite.agile import metrics
from perch_hoplite.db import interface as db_interface
import tensorflow as tf
import tqdm


@dataclasses.dataclass
class LinearClassifier:
  """Wrapper for linear classifier params and metadata."""

  beta: np.ndarray
  beta_bias: np.ndarray
  classes: tuple[str, ...]
  embedding_model_config: Any

  def __call__(self, embeddings: np.ndarray):
    return np.dot(embeddings, self.beta) + self.beta_bias

  def save(self, path: str):
    """Save the classifier to a path."""
    cfg = config_dict.ConfigDict()
    cfg.model_config = self.embedding_model_config
    cfg.classes = self.classes
    # Convert numpy arrays to base64 encoded blobs.
    beta_bytes = base64.b64encode(np.float32(self.beta).tobytes()).decode(
        'ascii'
    )
    beta_bias_bytes = base64.b64encode(
        np.float32(self.beta_bias).tobytes()
    ).decode('ascii')
    cfg.beta = beta_bytes
    cfg.beta_bias = beta_bias_bytes
    with open(path, 'w') as f:
      f.write(cfg.to_json())

  @classmethod
  def load(cls, path: str):
    """Load a classifier from a path."""
    with open(path, 'r') as f:
      cfg_json = json.loads(f.read())
      cfg = config_dict.ConfigDict(cfg_json)
    classes = cfg.classes
    beta = np.frombuffer(base64.b64decode(cfg.beta), dtype=np.float32)
    beta = np.reshape(beta, (-1, len(classes)))
    beta_bias = np.frombuffer(base64.b64decode(cfg.beta_bias), dtype=np.float32)
    embedding_model_config = cfg.model_config
    return cls(beta, beta_bias, classes, embedding_model_config)


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
) -> tuple[LinearClassifier, dict[str, float]]:
  """Train a linear classifier."""
  embedding_dim = data_manager.db.get_embedding_dim()
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

  for step, batch in enumerate(train_iter_):
    if step >= num_train_steps:
      break
    step_loss = train_step(
        batch.multihot, batch.embedding, batch.is_labeled_mask
    )
    progress.update(n=1)
    progress.set_description(f'Loss {step_loss:.8f}')
  progress.clear()
  progress.close()

  params = {
      'beta': lin_model.get_weights()[0],
      'beta_bias': lin_model.get_weights()[1],
  }
  eval_scores = eval_classifier(params, data_manager, eval_idxes)

  model_config = data_manager.db.get_metadata('model_config')
  linear_classifier = LinearClassifier(
      beta=params['beta'],
      beta_bias=params['beta_bias'],
      classes=data_manager.get_target_labels(),
      embedding_model_config=model_config,
  )
  return linear_classifier, eval_scores


@dataclasses.dataclass
class CsvWorkerState:
  """State for the CSV worker.

  Params:
    db: The base database from the parent thread.
    csv_filepath: The path to the CSV file to write.
    labels: The labels to write.
    threshold: The threshold for writing detections.
    _thread_db: The database to use in child threads.
  """

  db: db_interface.HopliteDBInterface
  csv_filepath: str
  labels: tuple[str, ...]
  threshold: float
  _thread_db: db_interface.HopliteDBInterface | None = None

  def get_thread_db(self) -> db_interface.HopliteDBInterface:
    if self._thread_db is None:
      self._thread_db = self.db.thread_split()
    return self._thread_db


def csv_worker_initializer(state: CsvWorkerState):
  """Initialize the CSV worker."""
  state.get_thread_db()
  with epath.Path(state.csv_filepath).open('w') as f:
    f.write('idx,project,filename,window_start,window_end,label,logits\n')


def csv_worker_fn(
    window_ids: np.ndarray, logits: np.ndarray, state: CsvWorkerState
) -> None:
  """Writes a CSV row for each detection.

  Args:
    window_ids: The window ids to write.
    logits: The logits for each window id.
    state: The state of the worker.
  """
  db = state.get_thread_db()
  with epath.Path(state.csv_filepath).open('a') as f:
    csv_writer = csv.writer(f)
    for idx, logit in zip(window_ids, logits):
      window = db.get_window(idx)
      recording = db.get_recording(window.recording_id)
      deployment = db.get_deployment(recording.deployment_id)
      for a in np.argwhere(logit > state.threshold):
        lbl = state.labels[a[0]]
        row = [
            idx,
            deployment.project,
            recording.filename,
            window.offsets[0],
            window.offsets[1],
            lbl,
            logit[a][0],
        ]
        row = [str(x) for x in row]
        csv_writer.writerow(row)


def batched_embedding_iterator(
    db: db_interface.HopliteDBInterface,
    window_ids: np.ndarray,
    batch_size: int = 1024,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
  """Iterate over embeddings in batches."""
  for q in range(0, len(window_ids), batch_size):
    batch_ids = window_ids[q : q + batch_size]
    batch_embs = db.get_embeddings_batch(batch_ids)
    yield batch_ids, batch_embs


def write_inference_csv(
    linear_classifier: LinearClassifier,
    db: db_interface.HopliteDBInterface,
    output_filepath: str,
    threshold: float,
    labels: Sequence[str] | None = None,
    window_ids: np.ndarray | Sequence[int] | None = None,
) -> None:
  """Write a CSV for all audio windows with logits above a threshold."""
  if window_ids is None:
    window_ids = np.array(db.match_window_ids())
  elif not isinstance(window_ids, np.ndarray):
    window_ids = np.array(window_ids)
  if labels is None:
    labels = linear_classifier.classes
  else:
    labels = tuple(set(labels).intersection(linear_classifier.classes))
  label_ids = {cl: i for i, cl in enumerate(linear_classifier.classes)}
  target_label_ids = np.array([label_ids[l] for l in labels])
  logits_fn = lambda batch_embs: linear_classifier(batch_embs)[
      :, target_label_ids
  ]
  detection_count = 0
  state = CsvWorkerState(
      db=db,
      csv_filepath=output_filepath,
      labels=labels,
      threshold=threshold,
  )
  emb_iter = batched_embedding_iterator(db, window_ids, batch_size=1024)
  with futures.ThreadPoolExecutor(
      max_workers=1,
      initializer=csv_worker_initializer,
      initargs=(state,),
  ) as executor:
    for batch_idxes, batch_embs in tqdm.tqdm(emb_iter):
      logits = logits_fn(batch_embs)
      # Filter out rows with no detections, avoiding extra database retrievals.
      detections = logits > threshold
      keep_rows = detections.max(axis=1)
      logits = logits[keep_rows]
      kept_idxes = batch_idxes[keep_rows]
      executor.submit(csv_worker_fn, kept_idxes, logits, state)
      detection_count += detections.sum()
  print(f'Wrote {detection_count} detections to {output_filepath}')
