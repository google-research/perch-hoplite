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

"""BirdCODE sound event detection model wrapper for Perch Hoplite."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
from typing import Any

import librosa
from ml_collections import config_dict
import numpy as np
from perch_hoplite.taxonomy import namespace
from perch_hoplite.zoo import hf_hub
from perch_hoplite.zoo import hf_models
from perch_hoplite.zoo import zoo_interface
import torch
import transformers

SAMPLE_RATE = 32_000
WINDOW_SECONDS = 5.0
_LOGITS_KEY = 'birdcode'
_TAXONOMY_NAMESPACE = 'gbif_backbone'
_N_FFT = 2048
_HOP_LENGTH = 256
_N_MELS = 256
_TOP_DB = 80.0
_NORM_MEAN = -13.369
_NORM_STD = 13.162
_EPS_CHANNELS = 2
_EPSILON = 1e-4
_INPUT_VECTOR_LENGTH = 64.0
_COSINE_MARGIN = 1.01
_CLASS_CHUNK_SIZE = 512
_CONVNEXT_BASE_DEPTHS = (3, 3, 27, 3)
_CONVNEXT_BASE_HIDDEN_SIZES = (128, 256, 512, 1024)


class _MelSpectrogram(torch.nn.Module):
  """Normalized log-mel spectrogram matching BirdCODE preprocessing."""

  window: torch.Tensor
  fb: torch.Tensor

  def __init__(self) -> None:
    super().__init__()
    fb = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=_N_FFT,
        n_mels=_N_MELS,
        fmin=0.0,
        fmax=float(SAMPLE_RATE // 2),
        htk=True,
        norm=None,
        dtype=np.float32,
    ).T
    self.register_buffer('window', torch.hann_window(_N_FFT))
    self.register_buffer('fb', torch.as_tensor(fb, dtype=torch.float32))

  def forward(self, waveform: torch.Tensor) -> torch.Tensor:
    """Converts waveform `[B, T]` to normalized mel `[B, 1, n_mels, frames]`."""
    stft = torch.stft(
        waveform,
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
        win_length=_N_FFT,
        window=self.window,
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = stft.abs().pow(2.0)
    mel = torch.matmul(self.fb.T, spec)
    mel_db = torch.log10(torch.clamp(mel, min=1e-10)) * 10.0
    mel_db = torch.maximum(
        mel_db, mel_db.amax(dim=(-2, -1), keepdim=True) - _TOP_DB
    )
    mel_norm = (mel_db - _NORM_MEAN) / _NORM_STD
    return mel_norm.unsqueeze(1)


def _append_eps_and_norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Appends AudioProtoPNet constant epsilon channels and computes norm."""
  eps = torch.full(
      (x.shape[0], _EPS_CHANNELS, x.shape[2], x.shape[3]),
      _EPSILON,
      device=x.device,
      dtype=x.dtype,
  )
  x_ext = torch.cat((x, eps), dim=1)
  norm = (
      torch.linalg.vector_norm(x_ext, dim=1, keepdim=True)
      .square_()
      .add_(_EPSILON)
      .sqrt_()
  )
  return x_ext, norm


def _normalize_prototypes(prototype_vectors: torch.Tensor) -> torch.Tensor:
  """Pre-normalizes prototype vectors for AudioProtoPNet cosine activation."""
  normalizing_factor = (
      prototype_vectors.shape[2] * prototype_vectors.shape[3]
  ) ** 0.5
  protos_ext, proto_norm = _append_eps_and_norm(prototype_vectors)
  # Upstream AudioProtoPNet adds `_EPSILON` outside sqrt on prototypes only,
  # while scaling features by `_INPUT_VECTOR_LENGTH` in `_normalize_features`.
  return protos_ext.div_((proto_norm + _EPSILON) * normalizing_factor)


def _normalize_features(
    features: torch.Tensor, proto_hw: tuple[int, int]
) -> torch.Tensor:
  """Normalizes spatial features for AudioProtoPNet cosine prototype matching."""
  normalizing_factor = (proto_hw[0] * proto_hw[1]) ** 0.5
  feat_ext, feat_norm = _append_eps_and_norm(features)
  return (_INPUT_VECTOR_LENGTH * feat_ext / feat_norm) / normalizing_factor


def _extract_submodel_state(
    state_dict: Mapping[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
  """Extracts checkpoint entries matching `prefix` with the prefix stripped."""
  return {
      k.removeprefix(prefix): v
      for k, v in state_dict.items()
      if k.startswith(prefix)
  }


class _PrototypeHead(torch.nn.Module):
  """BirdCODE prototype cosine-similarity and tempered-pooling head."""

  normalized_prototypes: torch.Tensor

  def __init__(
      self,
      normalized_prototypes: torch.Tensor,
      num_classes: int,
      pooling_temperature: float,
  ) -> None:
    super().__init__()
    num_prototypes = int(normalized_prototypes.shape[0])
    if num_prototypes % num_classes != 0:
      raise ValueError(
          f'num_prototypes ({num_prototypes}) must be divisible by '
          f'num_classes ({num_classes}).'
      )
    self.num_classes = num_classes
    self.prototypes_per_class = num_prototypes // num_classes
    self.pooling_temperature = pooling_temperature
    self.register_buffer(
        'normalized_prototypes', normalized_prototypes, persistent=False
    )
    self.weight = torch.nn.Parameter(
        torch.empty(num_classes, self.prototypes_per_class)
    )
    self.bias = torch.nn.Parameter(torch.empty(num_classes))

  @classmethod
  def from_state_dict(
      cls,
      state_dict: Mapping[str, torch.Tensor],
      num_prototypes: int,
      num_classes: int,
      pooling_temperature: float,
  ) -> _PrototypeHead:
    """Builds and loads the prototype head on CPU before device transfer."""
    raw_protos = state_dict['encoder.prototype_vectors']
    if int(raw_protos.shape[0]) != num_prototypes:
      raise ValueError(
          f'Expected {num_prototypes} prototypes, got {raw_protos.shape[0]}.'
      )
    with torch.no_grad():
      normalized_protos = _normalize_prototypes(raw_protos)
    head = cls(normalized_protos, num_classes, pooling_temperature)
    head.load_state_dict(
        _extract_submodel_state(state_dict, 'classifier.'), strict=True
    )
    with torch.no_grad():
      head.weight.relu_()
    return head

  def forward(self, hidden: torch.Tensor) -> torch.Tensor:
    """Computes tempered-pooled window log-odds in prototype chunks."""
    upsampled = torch.nn.functional.interpolate(
        hidden, scale_factor=2, mode='bilinear'
    )
    proto_hw = (
        int(self.normalized_prototypes.shape[2]),
        int(self.normalized_prototypes.shape[3]),
    )
    x_norm = _normalize_features(upsampled, proto_hw=proto_hw)
    max_cosine_scale = _INPUT_VECTOR_LENGTH * _COSINE_MARGIN

    logit_chunks: list[torch.Tensor] = []
    for c_start in range(0, self.num_classes, _CLASS_CHUNK_SIZE):
      c_end = min(c_start + _CLASS_CHUNK_SIZE, self.num_classes)
      p_start = c_start * self.prototypes_per_class
      p_end = c_end * self.prototypes_per_class
      proto_chunk = self.normalized_prototypes[p_start:p_end]
      pooled_chunk = (
          torch.relu(
              torch.nn.functional.conv2d(x_norm, proto_chunk).amax(dim=2)
          )
          / max_cosine_scale
      )
      reshaped = pooled_chunk.view(
          pooled_chunk.shape[0],
          c_end - c_start,
          self.prototypes_per_class,
          pooled_chunk.shape[-1],
      )
      cls_chunk = (
          torch.einsum('ncpw,cp->ncw', reshaped, self.weight[c_start:c_end])
          + self.bias[c_start:c_end, None]
      )
      logit_chunks.append(cls_chunk)

    subframe_logits = torch.cat(logit_chunks, dim=1)
    frame_probs = torch.sigmoid(subframe_logits)
    weights = (
        frame_probs
        if self.pooling_temperature == 1.0
        else torch.pow(frame_probs, self.pooling_temperature)
    )
    clip_probs = (weights * frame_probs).sum(dim=-1) / (
        weights.sum(dim=-1) + 1e-7
    )
    return torch.logit(torch.clamp(clip_probs, min=1e-7, max=1.0 - 1e-7))


def _build_mel_transform(
    state_dict: Mapping[str, torch.Tensor],
) -> _MelSpectrogram:
  """Constructs `_MelSpectrogram` and copies checkpoint buffers if present."""
  mel_transform = _MelSpectrogram()
  window_key = 'encoder.mel_transform.spectrogram.window'
  fb_key = 'encoder.mel_transform.mel_scale.fb'
  if window_key in state_dict:
    mel_transform.window.copy_(state_dict[window_key])
  if fb_key in state_dict:
    mel_transform.fb.copy_(state_dict[fb_key])
  return mel_transform


def _build_backbone(
    encoder_cfg: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]
) -> torch.nn.Module:
  """Constructs and loads the 1-channel ConvNeXt backbone."""
  backbone_cfg = transformers.ConvNextConfig(
      num_channels=1,
      depths=list(encoder_cfg.get('depths', _CONVNEXT_BASE_DEPTHS)),
      hidden_sizes=list(
          encoder_cfg.get('hidden_sizes', _CONVNEXT_BASE_HIDDEN_SIZES)
      ),
  )
  with hf_models.TORCH_LOAD_LOCK:
    backbone: torch.nn.Module = transformers.ConvNextModel(backbone_cfg)
  backbone.load_state_dict(
      _extract_submodel_state(state_dict, 'encoder.backbone.'), strict=True
  )
  return backbone


@dataclasses.dataclass
class BirdCODEModel(zoo_interface.EmbeddingModel):
  """Wrapper for Earth Species Project BirdCODE sound event detection model.

  Model card: https://huggingface.co/EarthSpeciesProject/sed-birdcode

  Extracts 1024-dimensional window embeddings by spatially mean-pooling the
  final ConvNeXt-Base feature map over each 5-second 32 kHz audio window, with
  optional species logit computation via the prototype classification head.

  Attributes:
    model_path: Local/remote directory or Hugging Face repo ID for BirdCODE.
    window_size_s: Window size in seconds for framing audio.
    hop_size_s: Hop size in seconds for framing audio.
    device: Device to run inference on ('cpu', 'gpu', 'cuda', or 'cuda:<idx>').
    compute_logits: Whether to also load and run the prototype classifier head.
    class_list: Loaded class list dictionary when `compute_logits` is True.
  """

  model_path: str
  window_size_s: float = WINDOW_SECONDS
  hop_size_s: float = WINDOW_SECONDS
  device: str = 'cpu'
  compute_logits: bool = False
  class_list: dict[str, namespace.ClassList] = dataclasses.field(
      default_factory=dict, init=False
  )
  _mel_transform: _MelSpectrogram = dataclasses.field(init=False, repr=False)
  _backbone: torch.nn.Module = dataclasses.field(init=False, repr=False)
  _head: _PrototypeHead | None = dataclasses.field(
      default=None, init=False, repr=False
  )

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> BirdCODEModel:
    """Loads BirdCODEModel from a configuration dictionary."""
    return cls(
        sample_rate=config.get('sample_rate', SAMPLE_RATE),
        model_path=config.model_path,
        window_size_s=config.get('window_size_s', WINDOW_SECONDS),
        hop_size_s=config.get('hop_size_s', WINDOW_SECONDS),
        device=config.get('device', 'cpu'),
        compute_logits=config.get('compute_logits', False),
    )

  def __post_init__(self) -> None:
    self.device = hf_models.resolve_torch_device(self.device)
    config_path = hf_hub.resolve_file(self.model_path, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
      birdcode_cfg: dict[str, Any] = json.load(f)

    expected_sr = int(birdcode_cfg.get('sample_rate', SAMPLE_RATE))
    if self.sample_rate != expected_sr:
      raise ValueError(
          f'BirdCODE expects sample_rate={expected_sr}, got {self.sample_rate}.'
      )
    labels: tuple[str, ...] = tuple(birdcode_cfg.get('labels', ()))
    if self.compute_logits and not labels:
      raise ValueError(
          "BirdCODE config must provide non-empty 'labels' when"
          ' compute_logits=True.'
      )

    ckpt_path = hf_hub.resolve_file(self.model_path, 'checkpoint/best_model.pt')
    state_dict: dict[str, torch.Tensor] = torch.load(
        ckpt_path, map_location='cpu', weights_only=True, mmap=True
    )
    encoder_cfg: Mapping[str, Any] = birdcode_cfg.get('encoder', {})

    self._mel_transform = (
        _build_mel_transform(state_dict).eval().to(self.device)
    )
    self._backbone = (
        _build_backbone(encoder_cfg, state_dict).eval().to(self.device)
    )
    if self.compute_logits:
      self.class_list = {
          _LOGITS_KEY: namespace.ClassList(_TAXONOMY_NAMESPACE, labels)
      }
      self._head = (
          _PrototypeHead.from_state_dict(
              state_dict=state_dict,
              num_prototypes=int(encoder_cfg['num_prototypes']),
              num_classes=len(labels),
              pooling_temperature=float(
                  birdcode_cfg.get('pooling_temperature', 1.0)
              ),
          )
          .eval()
          .to(self.device)
      )

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    """See base class."""
    return zoo_interface.embed_from_batch_embed_fn(
        self.batch_embed, audio_array
    )

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    """See base class."""
    framed = self.frame_audio(audio_batch, self.window_size_s, self.hop_size_s)
    batch_size, num_frames, window_samples = framed.shape
    flat_windows = np.ascontiguousarray(
        framed.reshape(batch_size * num_frames, window_samples),
        dtype=np.float32,
    )
    waveforms = torch.as_tensor(flat_windows, device=self.device)

    with torch.no_grad():
      mel = self._mel_transform(waveforms)
      hidden = self._backbone(pixel_values=mel).last_hidden_state
      window_embeds_t = hidden.mean(dim=(-2, -1))
      window_logits_t = self._head(hidden) if self._head is not None else None

    embeddings = (
        window_embeds_t.to('cpu').numpy().reshape(batch_size, num_frames, 1, -1)
    )
    logits: dict[str, np.ndarray] | None = None
    if window_logits_t is not None:
      logits = {
          _LOGITS_KEY: (
              window_logits_t.to('cpu')
              .numpy()
              .reshape(batch_size, num_frames, -1)
          )
      }
    return zoo_interface.InferenceOutputs(
        embeddings=embeddings, logits=logits, batched=True
    )
