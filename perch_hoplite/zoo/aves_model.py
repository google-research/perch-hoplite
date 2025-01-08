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

"""Zoo models using Jax."""

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Any

import jax
from jax import numpy as jnp
from jax import random
import jaxonnxruntime
from jaxonnxruntime.core import call_onnx
from jaxonnxruntime.core import handler
from jaxonnxruntime.core import onnx_node
from ml_collections.config_dict import config_dict
import numpy as np
from perch_hoplite.zoo import zoo_interface

import onnx


@handler.register_op('InstanceNormalization')
class InstanceNormalization(handler.Handler):
  """Implementation of the ONNX InstanceNormalization operator."""

  @classmethod
  def _prepare(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any], onnx_jax_impl: Any
  ):
    node.attrs_dict['epsilon'] = node.attrs.get('epsilon', 1e-5)

  @classmethod
  def version_6(
      cls, node: onnx_node.OnnxNode, inputs: Sequence[Any]
  ) -> Callable[..., Any]:
    """ONNX version_6 InstanceNormalization op."""
    cls._prepare(node, inputs, onnx_instancenormalization)
    return onnx_instancenormalization


@functools.partial(jax.jit, static_argnames=('epsilon',))
def onnx_instancenormalization(*input_args, epsilon: float):
  """https://github.com/onnx/onnx/blob/v1.12.0/docs/Operators.md#InstanceNormalization for more details."""
  input_, scale, b = input_args

  dims_input = len(input_.shape)
  input_mean = jnp.mean(
      input_, axis=tuple(range(dims_input))[2:], keepdims=True
  )
  input_var = jnp.var(input_, axis=tuple(range(dims_input))[2:], keepdims=True)

  dim_ones = (1,) * (dims_input - 2)
  scale = scale.reshape(-1, *dim_ones)
  b = b.reshape(-1, *dim_ones)

  return (input_ - input_mean) / jnp.sqrt(input_var + epsilon) * scale + b


@dataclasses.dataclass
class AVES(zoo_interface.EmbeddingModel):
  """Wrapper around AVES ONNX model.

  This model was originally trained to take audio with a 16 kHz sample rate.

  Each time the model gets called with a new input shape, a new JAX function
  gets created and compiled, and all parameters get copied. This could be
  slow.
  """

  model_path: str = ''
  model: onnx.onnx_ml_pb2.ModelProto = dataclasses.field(init=False)
  input_shape: tuple[int, ...] | None = dataclasses.field(
      default=None, init=False
  )
  model_func: Callable[[list[np.ndarray[Any, Any]]], list[jax.Array]] | None = (
      dataclasses.field(default=None, init=False)
  )

  @classmethod
  def from_config(
      cls, model_config: config_dict.ConfigDict
  ) -> zoo_interface.EmbeddingModel:
    return cls(**model_config)

  def __post_init__(self):
    jaxonnxruntime.config.update(
        'jaxort_only_allow_initializers_as_static_args', False
    )
    self.model = onnx.load(self.model_path)

  def embed(
      self, audio_array: np.ndarray[Any, Any]
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.embed_from_batch_embed_fn(
        self.batch_embed, audio_array
    )

  def batch_embed(
      self, audio_batch: np.ndarray[Any, Any]
  ) -> zoo_interface.InferenceOutputs:
    # Compile new function if necessary
    if audio_batch.shape != self.input_shape:
      key = random.PRNGKey(0)
      random_input = random.normal(key, audio_batch.shape, dtype=jnp.float32)
      model_func, model_params = call_onnx.call_onnx_model(
          self.model, [random_input]
      )
      self.input_shape = audio_batch.shape
      self.model_func = functools.partial(model_func, model_params)

    # Embed and add a single channel dimension
    embeddings = np.asarray(self.model_func([audio_batch])[0])
    embeddings = embeddings[:, :, np.newaxis, :]
    return zoo_interface.InferenceOutputs(embeddings=embeddings, batched=True)
