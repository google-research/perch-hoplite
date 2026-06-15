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

"""Implementations of inference interfaces for applying trained ONNX models."""

from __future__ import annotations

import dataclasses
from typing import Any

import kagglehub
from ml_collections import config_dict
import numpy as np
from perch_hoplite.taxonomy import namespace
from perch_hoplite.zoo import hf_hub
from perch_hoplite.zoo import zoo_interface


def _require_onnxruntime():
  try:
    import onnxruntime  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  except ImportError as e:
    raise ImportError(
        '`onnxruntime` is required to run OnnxModel but is not installed. '
        'Install it via the extra: pip install "perch-hoplite[onnx]".'
    ) from e
  return onnxruntime


@dataclasses.dataclass
class TaxonomyModelOnnx(zoo_interface.EmbeddingModel):
  """Taxonomy model backed by an ONNX Runtime session over a local file.

  Attributes:
    window_size_s: Analysis window length in seconds.
    hop_size_s: Hop size for inference.
    target_peak: Peak normalization value.
    input_name: ONNX input tensor name.
    output_map: Maps the roles 'embedding'/'logits'/'frontend' to ONNX output
      tensor names. Missing roles are skipped.
    model_path: Local path to the `.onnx` file (resolved by the caller).
    class_list: Loaded class_lists for the model's output logits.
  """

  window_size_s: float = 5.0
  hop_size_s: float = 5.0
  target_peak: float | None = 0.25
  input_name: str = 'inputs'
  output_map: dict[str, str] = dataclasses.field(default_factory=dict)
  model_path: str = ''
  class_list: dict[str, namespace.ClassList] = dataclasses.field(
      default_factory=dict
  )

  def __post_init__(self):
    if not self.model_path:
      raise ValueError('TaxonomyModelOnnx requires a local model_path.')
    ort = _require_onnxruntime()
    self._session = ort.InferenceSession(
        self.model_path, providers=['CPUExecutionProvider']
    )
    self._available = {o.name for o in self._session.get_outputs()}

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'TaxonomyModelOnnx':
    # Tolerate extra keys shared across preset configs.
    known = {f.name for f in dataclasses.fields(cls)}
    # Handle peak_norm -> target_peak rename if it comes from old config
    cfg = dict(config)
    if 'peak_norm' in cfg and 'target_peak' not in cfg:
      cfg['target_peak'] = cfg.pop('peak_norm')
    return cls(**{k: v for k, v in cfg.items() if k in known})

  @property
  def class_lists(self) -> dict[str, namespace.ClassList]:
    return self.class_list

  def _prepare(self, audio_array: np.ndarray) -> np.ndarray:
    framed = self.frame_audio(audio_array, self.window_size_s, self.hop_size_s)
    normalized = self.normalize_audio(framed, self.target_peak)
    return normalized

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    wanted = {
        role: name
        for role, name in self.output_map.items()
        if name in self._available
    }
    names = list(wanted.values())

    prepared_audio = self._prepare(audio_array)
    num_frames = prepared_audio.shape[0]

    results = dict(
        zip(
            names,
            self._session.run(names, {self.input_name: prepared_audio}),
        )
    )

    emb = results.get(wanted.get('embedding', ''))
    logits = results.get(wanted.get('logits', ''))
    spec = results.get(wanted.get('frontend', ''))

    embeddings = None
    if emb is not None:
      embeddings = np.asarray(emb).reshape(num_frames, 1, -1)

    logit_map = None
    if logits is not None:
      logit_map = {}
      logits_val = np.asarray(results[wanted['logits']]).reshape(num_frames, -1)
      if len(self.class_list) == 1:
        key = list(self.class_list.keys())[0]
        logit_map[key] = logits_val
      else:
        logit_map['label'] = logits_val

    frontend = None
    if spec is not None:
      spec = np.asarray(spec)
      frontend = spec.reshape(num_frames, *spec.shape[1:])

    return zoo_interface.InferenceOutputs(
        embeddings=embeddings,
        logits=logit_map,
        frontend=frontend,
        batched=False,
    )


class PerchV2OnnxModel(TaxonomyModelOnnx):
  """Perch ONNX model that automatically downloads weights and labels."""

  @classmethod
  def resolve_config(cls, config: config_dict.ConfigDict) -> dict[str, Any]:
    # Download weights from HF Hub
    model_path = hf_hub.download('justinchuby/Perch-onnx', 'perch_v2.onnx')

    # Download labels from Kaggle Hub
    try:
      csv_path = kagglehub.model_download(
          'google/bird-vocalization-classifier/tensorFlow2/perch_v2/2',
          path='assets/labels.csv',
      )
      with open(csv_path, 'r') as f:
        class_list_obj = namespace.ClassList.from_csv(f)
      class_list_dict = {class_list_obj.namespace: class_list_obj}
    except Exception:  # pylint: disable=broad-except
      class_list_dict = {}

    # Create config dict with resolved values
    cfg = dict(config)
    cfg['model_path'] = model_path
    cfg['class_list'] = class_list_dict
    return cfg

  @classmethod
  def from_config(cls, config: config_dict.ConfigDict) -> 'PerchV2OnnxModel':
    resolved_cfg = cls.resolve_config(config)
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in resolved_cfg.items() if k in known})
