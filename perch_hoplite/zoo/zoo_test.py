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

"""Tests for mass-embedding functionality."""

import numpy as np
from perch_hoplite.zoo import model_configs
from perch_hoplite.zoo import zoo_interface

from absl.testing import absltest
from absl.testing import parameterized


class ZooTest(parameterized.TestCase):

  def test_pooled_embeddings(self):
    outputs = zoo_interface.InferenceOutputs(
        embeddings=np.zeros([10, 2, 8]), batched=False
    )
    batched_outputs = zoo_interface.InferenceOutputs(
        embeddings=np.zeros([3, 10, 2, 8]), batched=True
    )

    # Check that no-op is no-op.
    non_pooled = outputs.pooled_embeddings('', '')
    self.assertSequenceEqual(non_pooled.shape, outputs.embeddings.shape)
    batched_non_pooled = batched_outputs.pooled_embeddings('', '')
    self.assertSequenceEqual(
        batched_non_pooled.shape, batched_outputs.embeddings.shape
    )

    for pooling_method in zoo_interface.POOLING_METHODS:
      if pooling_method == 'squeeze':
        # The 'squeeze' pooling method throws an exception if axis size is > 1.
        with self.assertRaises(ValueError):
          outputs.pooled_embeddings(pooling_method, '')
        continue
      elif pooling_method == 'flatten':
        # Concatenates over the target axis.
        time_pooled = outputs.pooled_embeddings(pooling_method, '')
        self.assertSequenceEqual(time_pooled.shape, [2, 80])
        continue

      time_pooled = outputs.pooled_embeddings(pooling_method, '')
      self.assertSequenceEqual(time_pooled.shape, [2, 8])
      batched_time_pooled = batched_outputs.pooled_embeddings(
          pooling_method, ''
      )
      self.assertSequenceEqual(batched_time_pooled.shape, [3, 2, 8])

      channel_pooled = outputs.pooled_embeddings('', pooling_method)
      self.assertSequenceEqual(channel_pooled.shape, [10, 8])
      batched_channel_pooled = batched_outputs.pooled_embeddings(
          '', pooling_method
      )
      self.assertSequenceEqual(batched_channel_pooled.shape, [3, 10, 8])

      both_pooled = outputs.pooled_embeddings(pooling_method, pooling_method)
      self.assertSequenceEqual(both_pooled.shape, [8])
      batched_both_pooled = batched_outputs.pooled_embeddings(
          pooling_method, pooling_method
      )
      self.assertSequenceEqual(batched_both_pooled.shape, [3, 8])

  def test_simple_model_configs(self):
    """Load check for configs without framework dependencies."""
    for model_config_name in [
        model_configs.ModelConfigName.PLACEHOLDER,
        model_configs.ModelConfigName.BEANS_BASELINE,
    ]:
      with self.subTest(model_config_name):
        preset_info = model_configs.get_preset_model_config(model_config_name)
        self.assertGreaterEqual(preset_info.embedding_dim, 0)

  def test_beans_baseline_model(self):
    """Load check for configs with framework dependencies."""
    model = model_configs.load_model_by_name(
        model_configs.ModelConfigName.BEANS_BASELINE
    )
    fake_audio = np.zeros([5 * 32000], dtype=np.float32)
    outputs = model.embed(fake_audio)
    self.assertSequenceEqual(outputs.embeddings.shape, [5, 1, 80])


if __name__ == '__main__':
  absltest.main()
