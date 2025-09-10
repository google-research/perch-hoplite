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

"""Wrapper for loading models from KaggleHub with backwards compatibility."""

import kagglehub
import tensorflow as tf

# Older model configs may contain a full model URL, as we used with TFHub.
# These are kept for back-compat, and converted to Kaggle Models slugs.
PERCH_TF_HUB_URL = (
    'https://www.kaggle.com/models/google/'
    'bird-vocalization-classifier/frameworks/TensorFlow2/'
    'variations/bird-vocalization-classifier/versions'
)
PERCH_V2_TF_HUB_URL = (
    'https://www.kaggle.com/models/google/bird-vocalization-classifier/'
    'tensorFlow2/perch_v2'
)
SURFPERCH_TF_HUB_URL = (
    'https://www.kaggle.com/models/google/surfperch/TensorFlow2'
)

PERCH_SLUG = 'google/bird-vocalization-classifier/tensorFlow2/'
BASE_KAGGLE_URL = 'https://www.kaggle.com/models/'

PERCH_V1_SLUG = PERCH_SLUG + 'bird-vocalization-classifier'
PERCH_V2_SLUG = PERCH_SLUG + 'perch_v2'
PERCH_V2_CPU_SLUG = PERCH_SLUG + 'perch_v2_cpu'
SURFPERCH_SLUG = 'google/surfperch/tensorFlow2/TensorFlow2/1'
HUMPBACK_SLUG = 'google/humpback-whale/tensorFlow2/humpback-whale'
MULTISPECIES_WHALE_SLUG = 'google/multispecies-whale/tensorFlow2/default'
YAMNET_SLUG = 'google/yamnet'
VGGISH_SLUG = 'google/vggish'


def normalize_slug(model_slug: str, model_version: int | None = None) -> str:
  """Convert the old full URLs used by TensorflowHub to KaggleHub slugs."""
  if model_slug == PERCH_TF_HUB_URL:
    return PERCH_V1_SLUG
  elif model_slug == PERCH_V2_TF_HUB_URL:
    return PERCH_V2_SLUG
  elif model_slug == SURFPERCH_TF_HUB_URL:
    return SURFPERCH_SLUG
  elif model_slug.startswith(BASE_KAGGLE_URL):
    return model_slug[len(BASE_KAGGLE_URL) :]

  if model_slug == PERCH_V1_SLUG and model_version in (5, 6, 7):
    # Due to SNAFUs uploading the new model version to KaggleModels,
    # some version numbers were skipped.
    raise ValueError('TFHub version 5, 6, and 7 do not exist.')
  if model_version is not None:
    return f'{model_slug}/{model_version}'
  return model_slug


def load(model_slug: str, model_version: int | None = None):
  """Download and load a model from KaggleHub."""
  if model_slug.startswith('/tmp'):
    # Assume this is a path to a downloaded model, rather than a kaggle model.
    return tf.saved_model.load(model_slug)

  model_path = normalize_slug(model_slug, model_version)
  cached_model_path = kagglehub.model_download(model_path)
  model = tf.saved_model.load(cached_model_path)
  return model


def resolve(model_slug: str, model_version: int | None = None) -> str:
  """Download a model from KaggleHub and return the cached model's path."""
  if model_slug.startswith('/tmp'):
    # Assume this is a path to a downloaded model, rather than a kaggle model.
    return model_slug
  model_path = normalize_slug(model_slug, model_version)
  cached_model_path = kagglehub.model_download(model_path)
  return cached_model_path
