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

"""Tests for linear classifier implementation."""

import os
import tempfile

from ml_collections import config_dict
import numpy as np
import pandas as pd
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
from perch_hoplite.db.tests import test_utils as db_test_utils

from absl.testing import absltest


class ClassifierTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def _make_linear_classifier(self, embedding_dim, classes):
    np.random.seed(1234)
    beta = np.float32(np.random.normal(size=(embedding_dim, len(classes))))
    beta_bias = np.float32(np.random.normal(size=(len(classes),)))
    embedding_model_config = config_dict.ConfigDict({
        'model_name': 'nelson',
    })
    return classifier.LinearClassifier(
        beta, beta_bias, classes, embedding_model_config
    )

  def test_call_linear_classifier(self):
    embedding_dim = 8
    classes = ('a', 'b', 'c')
    classy = self._make_linear_classifier(embedding_dim, classes)

    batch_embeddings = np.random.normal(size=(10, embedding_dim))
    predictions = classy(batch_embeddings)
    self.assertEqual(predictions.shape, (10, len(classes)))

    single_embedding = np.random.normal(size=(embedding_dim,))
    predictions = classy(single_embedding)
    self.assertEqual(predictions.shape, (len(classes),))

  def test_save_load_linear_classifier(self):
    embedding_dim = 8
    classes = ('a', 'b', 'c')
    classy = self._make_linear_classifier(embedding_dim, classes)
    classy_path = os.path.join(self.tempdir, 'classifier.json')
    classy.save(classy_path)
    classy_loaded = classifier.LinearClassifier.load(classy_path)
    np.testing.assert_allclose(classy_loaded.beta, classy.beta)
    np.testing.assert_allclose(classy_loaded.beta_bias, classy.beta_bias)
    self.assertSequenceEqual(classy_loaded.classes, classy.classes)
    self.assertEqual(classy_loaded.embedding_model_config.model_name, 'nelson')

  def test_train_linear_classifier(self):
    rng = np.random.default_rng(1234)
    embedding_dim = 8
    db = db_test_utils.make_db(
        path=self.tempdir,
        db_type='in_mem',
        num_embeddings=1024,
        rng=rng,
        embedding_dim=embedding_dim,
    )
    db_test_utils.add_random_labels(
        db, rng=rng, unlabeled_prob=0.5, positive_label_prob=0.1
    )
    data_manager = classifier_data.AgileDataManager(
        target_labels=db_test_utils.CLASS_LABELS,
        db=db,
        train_ratio=0.8,
        min_eval_examples=5,
        batch_size=32,
        weak_negatives_batch_size=10,
        rng=np.random.default_rng(42),
    )
    lc, eval_scores = classifier.train_linear_classifier(
        data_manager,
        learning_rate=0.01,
        weak_neg_weight=0.5,
        num_train_steps=128,
        loss='bce',
    )
    self.assertIsInstance(lc, classifier.LinearClassifier)
    np.testing.assert_equal(
        lc.beta.shape, (embedding_dim, len(db_test_utils.CLASS_LABELS))
    )
    np.testing.assert_equal(
        lc.beta_bias.shape, (len(db_test_utils.CLASS_LABELS),)
    )
    self.assertIn('roc_auc', eval_scores)

  def test_write_inference_csv(self):
    embedding_dim = 8
    rng = np.random.default_rng(1234)
    db = db_test_utils.make_db(
        path=self.tempdir,
        db_type='in_mem',
        num_embeddings=1024,
        rng=rng,
        embedding_dim=embedding_dim,
    )
    db_test_utils.add_random_labels(
        db, rng=rng, unlabeled_prob=0.5, positive_label_prob=0.1
    )
    classes = ('alpha', 'beta', 'delta', 'epsilon')
    classy = self._make_linear_classifier(embedding_dim, classes)
    inference_classes = ('alpha', 'epsilon', 'gamma')
    classy.beta_bias = 0.0
    csv_filepath = os.path.join(self.tempdir, 'inference.csv')
    classifier.write_inference_csv(
        embedding_ids=db.get_embedding_ids(),
        linear_classifier=classy,
        db=db,
        output_filepath=csv_filepath,
        threshold=0.0,
        labels=inference_classes,
    )
    inference_csv = pd.read_csv(csv_filepath)
    got_labels = np.unique(inference_csv['label'].values)
    # `gamma` is not in the inference_classes, so should not be in the output.
    expected_labels = ('alpha', 'epsilon')
    np.testing.assert_array_equal(got_labels, expected_labels)

    # We can estimate the total number of detections. There are 1024 embeddings,
    # and we will only have outputs for two classes. Each logit is > 0 with
    # probability 0.5, because we are using an unbiased random classifier
    # with random embeddings. So, we expect 1024 * 0.5 * 2 = 1024 detections.
    self.assertGreater(len(inference_csv), 1000)
    self.assertLess(len(inference_csv), 1050)

    # Spot check some of the inference scores.
    for i in range(16):
      emb_id = inference_csv['idx'][i]
      lbl = inference_csv['label'][i]
      got_logit = inference_csv['logits'][i]
      class_idx = classy.classes.index(lbl)
      embedding = db.get_embedding(emb_id)
      expect_logit = classy(embedding)[class_idx]
      self.assertEqual(np.float16(got_logit), np.float16(expect_logit))


if __name__ == '__main__':
  absltest.main()
