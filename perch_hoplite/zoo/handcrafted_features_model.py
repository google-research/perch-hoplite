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

"""Handcrafted features for linear models."""

import dataclasses

import librosa
from ml_collections import config_dict
import numpy as np
from perch_hoplite.zoo import zoo_interface


@dataclasses.dataclass
class HandcraftedFeaturesModel(zoo_interface.EmbeddingModel):
  """Wrapper for simple feature extraction.

  Parameters:
    window_size_s: The window size for framing the audio.
    hop_size_s: The hop size for framing the audio.
    features_config: The configuration for the features. Used as arguments to
      librosa.feature.melspectrogram or librosa.feature.mfcc, depending on
      `use_mfccs`.
    use_mfccs: Whether to use MFCCs or melspectrogram.
    aggregation: How to aggregate the features.
  """

  window_size_s: float
  hop_size_s: float
  features_config: config_dict.ConfigDict
  use_mfccs: bool = True
  aggregation: str = 'beans'

  @classmethod
  def from_config(
      cls, config: config_dict.ConfigDict
  ) -> 'HandcraftedFeaturesModel':
    return cls(**config)

  @classmethod
  def beans_baseline_config(
      cls, sample_rate=32000, frame_rate=100
  ) -> config_dict.ConfigDict:
    stride = sample_rate // frame_rate
    features_config = config_dict.ConfigDict({
        'sample_rate': sample_rate,
        'n_mfcc': 20,
        'hop_length': stride,
        'win_length': 2 * stride,
        'freq_range': (60.0, sample_rate / 2.0),
        'power': 2.0,
    })
    return config_dict.ConfigDict({
        'sample_rate': sample_rate,
        'features_config': features_config,
        'use_mfccs': True,
        'window_size_s': 1.0,
        'hop_size_s': 1.0,
        'aggregation': 'beans',
    })

  @classmethod
  def beans_baseline(
      cls, sample_rate=32000, frame_rate=100
  ) -> 'HandcraftedFeaturesModel':
    config = cls.beans_baseline_config(sample_rate, frame_rate)
    # pylint: disable=unexpected-keyword-arg
    return HandcraftedFeaturesModel.from_config(config)

  def compute_features(self, audio_array: np.ndarray) -> np.ndarray:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    specs = []
    for frame in framed_audio:
      if self.use_mfccs:
        features = librosa.feature.mfcc(
            y=frame,
            sr=self.sample_rate,
            hop_length=self.features_config.hop_length,
            win_length=self.features_config.win_length,
            center=True,
            norm='ortho',
            dct_type=2,
            n_mfcc=self.features_config.n_mfcc,
        )
      else:
        features = librosa.feature.melspectrogram(
            y=frame,
            sr=self.sample_rate,
            hop_length=self.features_config.hop_length,
            win_length=self.features_config.win_length,
            center=True,
            n_mels=self.features_config.n_mels,
            power=self.features_config.power,
        )
      specs.append(features)
    return np.stack(specs, axis=0)

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    # Melspecs will have shape [melspec_channels, frames]
    features = self.compute_features(audio_array)
    print(f'features: {features.shape}')
    if self.aggregation == 'beans':
      features = np.concatenate(
          [
              features.mean(axis=-1),
              features.std(axis=-1),
              features.min(axis=-1),
              features.max(axis=-1),
          ],
          axis=-1,
      )
    else:
      raise ValueError(f'unrecognized aggregation: {self.aggregation}')
    # Add a trivial channels dimension.
    features = features[:, np.newaxis, :]
    return zoo_interface.InferenceOutputs(features, None, None)

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)
