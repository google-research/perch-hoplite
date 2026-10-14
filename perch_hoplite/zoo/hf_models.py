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

"""Hugging Face model wrappers for Perch Hoplite."""

import dataclasses
import os
import tempfile
import threading
from typing import Any

from etils import epath
import librosa
from ml_collections import config_dict
import numpy as np
from perch_hoplite.zoo import zoo_interface
from PIL import Image
import torch
import transformers

_LOAD_LOCK = threading.Lock()


def _load_pretrained(
    model_path: str, model_cls: Any, processor_cls: Any
) -> tuple[Any, Any]:
  """Loads a Hugging Face model and processor thread-safely."""
  with _LOAD_LOCK:
    processor = processor_cls.from_pretrained(model_path)
    model = model_cls.from_pretrained(model_path)
  return model, processor


@dataclasses.dataclass
class CLAPModel(zoo_interface.EmbeddingModel):
  """Wrapper for Hugging Face CLAP model.

  Model card: https://huggingface.co/laion/clap-htsat-unfused

  Attributes:
    model_path: Path or Hugging Face repo ID for model files.
    device: Device to run inference on ("cpu" or "gpu").
    model: Loaded CLAP model.
    processor: Loaded CLAP processor.
  """

  model_path: str
  device: str = "cpu"
  model: transformers.ClapModel = dataclasses.field(init=False)
  processor: Any = dataclasses.field(init=False)

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> "CLAPModel":
    return cls(
        sample_rate=config.sample_rate,
        model_path=config.model_path,
        device=config.get("device", "cpu"),
    )

  def __post_init__(self) -> None:
    self.device = (
        "cuda" if torch.cuda.is_available() and self.device == "gpu" else "cpu"
    )
    self.model, self.processor = _load_pretrained(
        self.model_path, transformers.ClapModel, transformers.ClapProcessor
    )
    self.model.eval()
    self.model.to(self.device)

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    return zoo_interface.embed_from_batch_embed_fn(
        self.batch_embed, audio_array
    )

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    waveforms = list(audio_batch)
    inputs = self.processor(
        audios=waveforms,
        sampling_rate=self.sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    with torch.no_grad():
      audio_embeds = self.model.get_audio_features(**inputs)
      audio_embeds = audio_embeds.to("cpu").numpy()

    # Shape: [Batch, Time, Channels, Features]
    # CLAP outputs one embedding per audio clip, so Time=1, Channels=1.
    embeddings = audio_embeds[:, np.newaxis, np.newaxis, :]
    return zoo_interface.InferenceOutputs(embeddings=embeddings, batched=True)


@dataclasses.dataclass
class CLIPModel(zoo_interface.EmbeddingModel):
  """Wrapper for Hugging Face CLIP model (applied to spectrograms).

  Model card: https://huggingface.co/openai/clip-vit-base-patch32

  Attributes:
    model_path: Path or Hugging Face repo ID for model files.
    device: Device to run inference on ("cpu" or "gpu").
    model: Loaded CLIP model.
    processor: Loaded CLIP processor.
    n_mels: Number of Mel bins for spectrogram.
    hop_length: Hop length for spectrogram.
    n_fft: FFT size for spectrogram.
    fmin: Minimum frequency for spectrogram.
    fmax: Maximum frequency for spectrogram.
  """

  model_path: str
  device: str = "cpu"
  model: transformers.CLIPModel = dataclasses.field(init=False)
  processor: Any = dataclasses.field(init=False)
  # Spectrogram parameters (defaulting to perch_8 settings)
  n_mels: int = 128
  hop_length: int = 320
  n_fft: int = 1024
  fmin: float = 60.0
  fmax: float = 16000.0

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> "CLIPModel":
    return cls(
        sample_rate=config.sample_rate,
        model_path=config.model_path,
        device=config.get("device", "cpu"),
        n_mels=config.get("n_mels", 128),
        hop_length=config.get("hop_length", 320),
        n_fft=config.get("n_fft", 1024),
        fmin=config.get("fmin", 60.0),
        fmax=config.get("fmax", 16000.0),
    )

  def __post_init__(self) -> None:
    self.device = (
        "cuda" if torch.cuda.is_available() and self.device == "gpu" else "cpu"
    )
    self.model, self.processor = _load_pretrained(
        self.model_path, transformers.CLIPModel, transformers.CLIPProcessor
    )
    self.model.eval()
    self.model.to(self.device)

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    return zoo_interface.embed_from_batch_embed_fn(
        self.batch_embed, audio_array
    )

  def _audio_to_image(self, audio: np.ndarray) -> Image.Image:
    # Compute Mel Spectrogram using librosa with perch_8 settings
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=self.sample_rate,
        n_mels=self.n_mels,
        hop_length=self.hop_length,
        n_fft=self.n_fft,
        fmin=self.fmin,
        fmax=self.fmax,
        power=1.0,
    )
    # Apply Log Scaling: 0.1 * log(max(x, 1e-5))
    log_spec = 0.1 * np.log(np.maximum(mel_spec, 1e-5))

    # Normalize to 0-255
    min_val, max_val = log_spec.min(), log_spec.max()
    if max_val > min_val:
      normalized = ((log_spec - min_val) / (max_val - min_val) * 255).astype(
          np.uint8
      )
    else:
      normalized = np.zeros_like(log_spec, dtype=np.uint8)

    # Replicate across 3 channels
    rgb_data = np.stack([normalized, normalized, normalized], axis=-1)

    image = Image.fromarray(rgb_data)
    image = image.resize((224, 224))
    return image

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    images = [self._audio_to_image(audio) for audio in audio_batch]
    inputs = self.processor(
        images=images,
        return_tensors="pt",
    )
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    with torch.no_grad():
      image_embeds = self.model.get_image_features(**inputs)
      image_embeds = image_embeds.to("cpu").numpy()

    # Shape: [Batch, Time, Channels, Features]
    embeddings = image_embeds[:, np.newaxis, np.newaxis, :]
    return zoo_interface.InferenceOutputs(embeddings=embeddings, batched=True)
