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

import importlib
import importlib.util
import json
from unittest import mock

from ml_collections import config_dict
import numpy as np
from perch_hoplite.zoo import hf_hub
from perch_hoplite.zoo import model_configs
from perch_hoplite.zoo import zoo_interface

from absl.testing import absltest
from absl.testing import parameterized

_HAS_TORCH = (
    importlib.util.find_spec('torch') is not None
    and importlib.util.find_spec('transformers') is not None
)


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

  @absltest.skipIf(not _HAS_TORCH, 'torch and transformers not installed')
  @mock.patch('transformers.ClapProcessor.from_pretrained')
  @mock.patch('transformers.ClapModel.from_pretrained')
  def test_clap_model(self, mock_model_load, mock_processor_load):
    mock_processor = mock.Mock()
    mock_model = mock.Mock()
    mock_processor_load.return_value = mock_processor
    mock_model_load.return_value = mock_model

    # Mock processor output
    mock_tensor = mock.Mock()
    mock_processor.return_value = {'input_features': mock_tensor}
    mock_tensor.to.return_value = mock_tensor
    # Mock model output: get_audio_features
    mock_audio_embeds = mock.Mock()
    mock_model.get_audio_features.return_value = mock_audio_embeds
    mock_audio_embeds.to.return_value.numpy.return_value = np.zeros([2, 512])

    config = config_dict.ConfigDict({
        'sample_rate': 48000,
        'model_path': 'placeholder_path',
    })
    model = model_configs.get_model_class('clap').from_config(config)
    fake_audio = np.zeros([2, 48000 * 5], dtype=np.float32)
    outputs = model.batch_embed(fake_audio)

    self.assertEqual(outputs.embeddings.shape[0], 2)
    self.assertEqual(outputs.embeddings.shape[1], 1)
    self.assertEqual(outputs.embeddings.shape[2], 1)
    self.assertEqual(outputs.embeddings.shape[3], 512)

  @absltest.skipIf(not _HAS_TORCH, 'torch and transformers not installed')
  @mock.patch('transformers.CLIPProcessor.from_pretrained')
  @mock.patch('transformers.CLIPModel.from_pretrained')
  @mock.patch('librosa.feature.melspectrogram')
  def test_clip_model(self, mock_melspec, mock_model_load, mock_processor_load):
    mock_processor = mock.Mock()
    mock_model = mock.Mock()
    mock_processor_load.return_value = mock_processor
    mock_model_load.return_value = mock_model

    # Mock librosa melspectrogram
    mock_melspec.return_value = np.zeros([128, 500])

    # Mock processor output
    mock_tensor = mock.Mock()
    mock_processor.return_value = {'pixel_values': mock_tensor}
    mock_tensor.to.return_value = mock_tensor

    # Mock model output: get_image_features
    mock_image_embeds = mock.Mock()
    mock_model.get_image_features.return_value = mock_image_embeds
    mock_image_embeds.to.return_value.numpy.return_value = np.zeros([2, 512])

    config = config_dict.ConfigDict({
        'sample_rate': 32000,
        'model_path': 'placeholder_path',
    })
    model = model_configs.get_model_class('clip').from_config(config)
    fake_audio = np.zeros([2, 32000 * 5], dtype=np.float32)
    outputs = model.batch_embed(fake_audio)

    self.assertEqual(outputs.embeddings.shape[0], 2)
    self.assertEqual(outputs.embeddings.shape[1], 1)
    self.assertEqual(outputs.embeddings.shape[2], 1)
    self.assertEqual(outputs.embeddings.shape[3], 512)

  @mock.patch('huggingface_hub.hf_hub_download')
  def test_hf_hub_resolve_and_stage(self, mock_download):
    temp_dir = self.create_tempdir()
    local_cfg = temp_dir.create_file('config.json', content='{}').full_path

    self.assertEqual(
        hf_hub.stage_local_dir(temp_dir.full_path), temp_dir.full_path
    )
    self.assertEqual(hf_hub.stage_local_dir('org/repo'), 'org/repo')
    self.assertEqual(
        hf_hub.resolve_file(temp_dir.full_path, 'config.json'), local_cfg
    )
    mock_download.assert_not_called()

    with self.assertRaises(FileNotFoundError):
      hf_hub.stage_local_dir(temp_dir.full_path + '_missing')
    with self.assertRaises(FileNotFoundError):
      hf_hub.resolve_file(temp_dir.full_path, 'missing.json')

    mock_download.return_value = '/cached/config.json'
    self.assertEqual(
        hf_hub.resolve_file('org/repo', 'config.json'), '/cached/config.json'
    )
    mock_download.assert_called_once_with(
        repo_id='org/repo', filename='config.json'
    )

  def test_birdcode_preset_config(self):
    preset_info = model_configs.get_preset_model_config(
        model_configs.ModelConfigName.BIRDCODE
    )
    self.assertEqual(preset_info.preset_name, 'birdcode')
    self.assertEqual(preset_info.model_key, 'birdcode')
    self.assertEqual(preset_info.embedding_dim, 1024)
    self.assertEqual(preset_info.model_config.sample_rate, 32000)
    self.assertEqual(preset_info.model_config.window_size_s, 5.0)
    self.assertEqual(preset_info.model_config.hop_size_s, 5.0)

  @absltest.skipIf(not _HAS_TORCH, 'torch and transformers not installed')
  def test_birdcode_model(self):
    torch = importlib.import_module('torch')
    transformers = importlib.import_module('transformers')

    temp_dir = self.create_tempdir()
    ckpt_dir = temp_dir.mkdir('checkpoint')

    labels = [
        'Apteryx owenii',
        'Nestor notabilis',
        'Prosthemadera novaeseelandiae',
    ]
    depths = [1, 1, 1, 1]
    hidden_sizes = [8, 16, 32, 64]
    embed_dim = hidden_sizes[-1]
    num_prototypes = 6
    temp_dir.create_file(
        'config.json',
        content=json.dumps({
            'labels': labels,
            'sample_rate': 32000,
            'target_duration': 5.0,
            'encoder': {
                'type': 'audioprotopnet_frame',
                'num_prototypes': num_prototypes,
                'depths': depths,
                'hidden_sizes': hidden_sizes,
            },
        }),
    )

    backbone_cfg = transformers.ConvNextConfig(
        num_channels=1, depths=depths, hidden_sizes=hidden_sizes
    )
    backbone = transformers.ConvNextModel(backbone_cfg)
    state_dict = {
        f'encoder.backbone.{k}': v for k, v in backbone.state_dict().items()
    }
    state_dict['encoder.prototype_vectors'] = torch.zeros(
        num_prototypes, embed_dim, 1, 1
    )
    state_dict['classifier.weight'] = torch.ones(
        len(labels), num_prototypes // len(labels)
    )
    state_dict['classifier.bias'] = torch.zeros(len(labels))
    torch.save(state_dict, ckpt_dir.create_file('best_model.pt').full_path)

    config = config_dict.ConfigDict({
        'sample_rate': 32000,
        'model_path': temp_dir.full_path,
        'window_size_s': 5.0,
        'hop_size_s': 5.0,
    })
    model = model_configs.get_model_class('birdcode').from_config(config)
    self.assertEqual(model.class_list, {})
    fake_batch = np.zeros([2, 32000 * 5], dtype=np.float32)
    outputs = model.batch_embed(fake_batch)
    self.assertSequenceEqual(outputs.embeddings.shape, [2, 1, 1, embed_dim])
    self.assertIsNone(outputs.logits)

    single_outputs = model.embed(np.zeros([32000 * 10], dtype=np.float32))
    self.assertSequenceEqual(single_outputs.embeddings.shape, [2, 1, embed_dim])

    config_with_logits = config_dict.ConfigDict({
        'sample_rate': 32000,
        'model_path': temp_dir.full_path,
        'window_size_s': 5.0,
        'hop_size_s': 5.0,
        'compute_logits': True,
    })
    logits_model = model_configs.get_model_class('birdcode').from_config(
        config_with_logits
    )
    logits_outputs = logits_model.batch_embed(fake_batch)
    self.assertSequenceEqual(
        logits_outputs.embeddings.shape, [2, 1, 1, embed_dim]
    )
    self.assertIn('birdcode', logits_outputs.logits)
    self.assertSequenceEqual(logits_outputs.logits['birdcode'].shape, [2, 1, 3])
    self.assertEqual(
        logits_model.class_list['birdcode'].namespace, 'gbif_backbone'
    )
    self.assertEqual(logits_model.class_list['birdcode'].classes, tuple(labels))


if __name__ == '__main__':
  absltest.main()
