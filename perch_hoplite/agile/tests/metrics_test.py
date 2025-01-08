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

"""Tests for metrics."""

import numpy as np
from perch_hoplite.agile import metrics
from absl.testing import absltest


class MetricsTest(absltest.TestCase):

  def test_average_precision_no_labels(self):
    batch_size = 4
    num_classes = 5

    np.random.seed(42)
    logits = np.random.normal(size=[batch_size, num_classes])
    labels = np.zeros_like(logits)
    av_prec = np.mean(metrics.average_precision(logits, labels))
    self.assertEqual(av_prec, 0.0)


if __name__ == "__main__":
  absltest.main()
