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

"""Tests for mass-embedding functionality."""

from ml_collections import config_dict
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

  def test_logits_output_head_model_type(self):
    class_list = zoo_interface.namespace.ClassList('fake', ['alpha', 'beta'])
    # Valid defaults
    head = zoo_interface.LogitsOutputHead(
        model_path='/dev/null',
        logits_key='output_head',
        logits_model=None,
        class_list=class_list,
    )
    self.assertEqual(head.model_type, 'tf_saved_model')

    # Instantiating with invalid model type raises ValueError
    with self.assertRaisesRegex(ValueError, 'Unknown model type'):
      zoo_interface.LogitsOutputHead(
          model_path='/dev/null',
          logits_key='output_head',
          logits_model=None,
          class_list=class_list,
          model_type='unrecognized_type',
      )

    # Test from_config with invalid model_type
    config = config_dict.ConfigDict({
        'model_path': '/dev/null',
        'logits_key': 'output_head',
        'model_type': 'unrecognized_type',
    })
    with self.assertRaisesRegex(ValueError, 'Unknown model type'):
      zoo_interface.LogitsOutputHead.from_config(config)

    # Bypassing __post_init__ to test save_model validation
    head.model_type = 'unrecognized_type'
    with self.assertRaisesRegex(ValueError, 'Unknown model type'):
      head.save_model('/dev/null', '')


if __name__ == '__main__':
  absltest.main()
