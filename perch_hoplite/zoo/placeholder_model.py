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

"""Implementations of inference interfaces for applying trained models."""

import dataclasses

from ml_collections import config_dict
import numpy as np
from perch_hoplite.taxonomy import namespace_db
from perch_hoplite.zoo import zoo_interface


@dataclasses.dataclass
class PlaceholderModel(zoo_interface.EmbeddingModel):
  """Test implementation of the EmbeddingModel zoo_interface."""

  embedding_size: int = 128
  make_embeddings: bool = True
  make_logits: bool = True
  make_separated_audio: bool = True
  make_frontend: bool = True
  do_frame_audio: bool = False
  window_size_s: float = 1.0
  hop_size_s: float = 1.0
  frontend_size: tuple[int, int] = (32, 32)

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'PlaceholderModel':
    return cls(**config)

  def __post_init__(self):
    db = namespace_db.load_db()
    self.class_list = db.class_lists['caples']

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    outputs = {}
    if self.do_frame_audio:
      audio_array = self.frame_audio(
          audio_array, self.window_size_s, self.hop_size_s
      )
    time_size = audio_array.shape[0] // int(
        self.window_size_s * self.sample_rate
    )
    if self.make_embeddings:
      outputs['embeddings'] = np.zeros(
          [time_size, 1, self.embedding_size], np.float32
      )
    if self.make_frontend:
      outputs['frontend'] = np.zeros(
          [time_size, self.frontend_size[0], self.frontend_size[1]], np.float32
      )
    if self.make_logits:
      outputs['logits'] = {
          'label': np.zeros(
              [time_size, len(self.class_list.classes)], np.float32
          ),
          'other_label': np.ones(
              [time_size, len(self.class_list.classes)], np.float32
          ),
      }
    if self.make_separated_audio:
      outputs['separated_audio'] = np.zeros(
          [2, audio_array.shape[-1]], np.float32
      )
    return zoo_interface.InferenceOutputs(**outputs)

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)
