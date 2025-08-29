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

"""Tests for mass-embedding functionality."""

import os
import tempfile

from ml_collections import config_dict
import numpy as np
from perch_hoplite.taxonomy import namespace
from perch_hoplite.zoo import models_tf
from perch_hoplite.zoo import placeholder_model
from perch_hoplite.zoo import taxonomy_model_tf
import tensorflow as tf

from absl.testing import absltest
from absl.testing import parameterized


class ZooTest(parameterized.TestCase):

  def test_sep_embed_wrapper(self):
    """Check that the joint-model wrapper works as intended."""
    separator = placeholder_model.PlaceholderModel(
        sample_rate=22050,
        make_embeddings=False,
        make_logits=False,
        make_separated_audio=True,
    )

    embeddor = placeholder_model.PlaceholderModel(
        sample_rate=22050,
        make_embeddings=True,
        make_logits=True,
        make_separated_audio=False,
    )
    fake_config = config_dict.ConfigDict()
    sep_embed = models_tf.SeparateEmbedModel(
        sample_rate=22050,
        taxonomy_model_tf_config=fake_config,
        separator_model_tf_config=fake_config,
        separation_model=separator,
        embedding_model=embeddor,
    )
    audio = np.zeros(5 * 22050, np.float32)

    outputs = sep_embed.embed(audio)
    # The PlaceholderModel produces one embedding per second, and we have
    # five seconds of audio, with two separated channels, plus the channel
    # for the raw audio.
    # Note that this checks that the sample-rate conversion between the
    # separation model and embedding model has worked correctly.
    self.assertSequenceEqual(
        outputs.embeddings.shape, [5, 3, embeddor.embedding_size]
    )
    # The Sep+Embed model takes the max logits over the channel dimension.
    self.assertSequenceEqual(
        outputs.logits['label'].shape, [5, len(embeddor.class_list.classes)]
    )

  @parameterized.product(
      model_return_type=('tuple', 'dict'),
      batchable=(True, False),
  )
  def test_taxonomy_model_tf(self, model_return_type, batchable):
    class FakeModelFn:
      output_depths = {'label': 3, 'embedding': 256}

      def infer_tf(self, audio_array):
        outputs = {
            k: np.zeros([audio_array.shape[0], d], dtype=np.float32)
            for k, d in self.output_depths.items()
        }
        if model_return_type == 'tuple':
          # Published Perch models v1 through v4 returned a tuple, not a dict.
          return outputs['label'], outputs['embedding']
        return outputs

    class_list = {
        'label': namespace.ClassList('fake', ['alpha', 'beta', 'delta'])
    }
    wrapped_model = taxonomy_model_tf.TaxonomyModelTF(
        sample_rate=32000,
        model_path='/dev/null',
        window_size_s=5.0,
        hop_size_s=5.0,
        model=FakeModelFn(),
        class_list=class_list,
        batchable=batchable,
    )

    # Check that a single frame of audio is handled properly.
    outputs = wrapped_model.embed(np.zeros([5 * 32000], dtype=np.float32))
    self.assertFalse(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [1, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [1, 3])

    # Check that multi-frame audio is handled properly.
    outputs = wrapped_model.embed(np.zeros([20 * 32000], dtype=np.float32))
    self.assertFalse(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [4, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [4, 3])

    # Check that a batch of single frame of audio is handled properly.
    outputs = wrapped_model.batch_embed(
        np.zeros([10, 5 * 32000], dtype=np.float32)
    )
    self.assertTrue(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [10, 1, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [10, 1, 3])

    # Check that a batch of multi-frame audio is handled properly.
    outputs = wrapped_model.batch_embed(
        np.zeros([2, 20 * 32000], dtype=np.float32)
    )
    self.assertTrue(outputs.batched)
    self.assertSequenceEqual(outputs.embeddings.shape, [2, 4, 1, 256])
    self.assertSequenceEqual(outputs.logits['label'].shape, [2, 4, 3])

if __name__ == '__main__':
  absltest.main()
