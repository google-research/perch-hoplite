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

"""Tests for Kaggle model URL normalization and version forwarding."""

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from perch_hoplite.zoo import kaggle_hub


_URL_CASES = (
    ('perch_v1', kaggle_hub.PERCH_TF_HUB_URL, kaggle_hub.PERCH_V1_SLUG),
    ('perch_v2', kaggle_hub.PERCH_V2_TF_HUB_URL, kaggle_hub.PERCH_V2_SLUG),
    ('surfperch', kaggle_hub.SURFPERCH_TF_HUB_URL, kaggle_hub.SURFPERCH_SLUG),
    (
        'generic',
        kaggle_hub.BASE_KAGGLE_URL + kaggle_hub.HUMPBACK_SLUG,
        kaggle_hub.HUMPBACK_SLUG,
    ),
    (
        'capitalized_framework',
        kaggle_hub.BASE_KAGGLE_URL
        + 'google/bird-vocalization-classifier/TensorFlow2/'
        + 'bird-vocalization-classifier',
        'google/bird-vocalization-classifier/TensorFlow2/'
        + 'bird-vocalization-classifier',
    ),
)


class KaggleHubTest(parameterized.TestCase):

  @parameterized.named_parameters(*_URL_CASES)
  def test_url_preserves_requested_version(self, url, slug):
    self.assertEqual(kaggle_hub.normalize_slug(url, 2), f'{slug}/2')

  @parameterized.named_parameters(*_URL_CASES)
  def test_url_without_version_keeps_existing_behavior(self, url, slug):
    self.assertEqual(kaggle_hub.normalize_slug(url), slug)

  @parameterized.parameters(
      kaggle_hub.PERCH_V1_SLUG,
      kaggle_hub.PERCH_V2_SLUG,
      kaggle_hub.SURFPERCH_SLUG,
  )
  def test_slug_preserves_requested_version(self, slug):
    self.assertEqual(kaggle_hub.normalize_slug(slug, 2), f'{slug}/2')

  @parameterized.product(
      slug=(kaggle_hub.PERCH_V1_SLUG, kaggle_hub.PERCH_V2_SLUG),
      prefix=('', kaggle_hub.BASE_KAGGLE_URL),
  )
  def test_embedded_version_without_separate_argument_is_unchanged(
      self, slug, prefix
  ):
    self.assertEqual(
        kaggle_hub.normalize_slug(f'{prefix}{slug}/2'), f'{slug}/2'
    )

  @parameterized.parameters('', kaggle_hub.BASE_KAGGLE_URL)
  def test_matching_embedded_and_explicit_versions_are_not_duplicated(
      self, prefix
  ):
    slug = kaggle_hub.PERCH_V2_SLUG
    self.assertEqual(
        kaggle_hub.normalize_slug(f'{prefix}{slug}/2', 2), f'{slug}/2'
    )

  @parameterized.parameters('', kaggle_hub.BASE_KAGGLE_URL)
  def test_conflicting_embedded_and_explicit_versions_are_rejected(
      self, prefix
  ):
    with self.assertRaisesRegex(ValueError, 'conflicts'):
      kaggle_hub.normalize_slug(f'{prefix}{kaggle_hub.PERCH_V2_SLUG}/2', 3)

  @parameterized.parameters('load', 'resolve')
  def test_conflicting_versions_do_not_download(self, method):
    source = kaggle_hub.BASE_KAGGLE_URL + kaggle_hub.PERCH_V2_SLUG + '/2'
    with mock.patch.dict(sys.modules, {'tensorflow': mock.MagicMock()}):
      with mock.patch.object(
          kaggle_hub.kagglehub, 'model_download'
      ) as download:
        with self.assertRaisesRegex(ValueError, 'conflicts'):
          getattr(kaggle_hub, method)(source, 3)
    download.assert_not_called()

  @parameterized.product(
      source=(kaggle_hub.PERCH_V1_SLUG, kaggle_hub.PERCH_TF_HUB_URL),
      version=(5, 6, 7),
  )
  def test_invalid_perch_v1_versions_are_rejected(self, source, version):
    with self.assertRaisesRegex(ValueError, 'do not exist'):
      kaggle_hub.normalize_slug(source, version)

  @parameterized.named_parameters(*_URL_CASES)
  def test_resolve_forwards_version_and_file_path(self, url, slug):
    with mock.patch.object(
        kaggle_hub.kagglehub, 'model_download', return_value='/cache/model'
    ) as download:
      result = kaggle_hub.resolve(url, 2, path='assets/labels.csv')
    self.assertEqual(result, '/cache/model')
    download.assert_called_once_with(f'{slug}/2', path='assets/labels.csv')

  @parameterized.named_parameters(*_URL_CASES)
  def test_load_forwards_version_to_downloader(self, url, slug):
    # Isolate the optional framework import; no weights or network are needed.
    tensorflow = mock.MagicMock()
    with mock.patch.dict(sys.modules, {'tensorflow': tensorflow}):
      with mock.patch.object(
          kaggle_hub.kagglehub, 'model_download', return_value='/cache/model'
      ) as download:
        result = kaggle_hub.load(url, 2)
    download.assert_called_once_with(f'{slug}/2')
    tensorflow.saved_model.load.assert_called_once_with('/cache/model')
    self.assertIs(result, tensorflow.saved_model.load.return_value)

  def test_local_resolve_does_not_download(self):
    with mock.patch.object(kaggle_hub.kagglehub, 'model_download') as download:
      self.assertEqual(
          kaggle_hub.resolve('/tmp/local-model', 2), '/tmp/local-model'
      )
    download.assert_not_called()

  def test_local_load_does_not_download(self):
    tensorflow = mock.MagicMock()
    with mock.patch.dict(sys.modules, {'tensorflow': tensorflow}):
      with mock.patch.object(
          kaggle_hub.kagglehub, 'model_download'
      ) as download:
        result = kaggle_hub.load('/tmp/local-model', 2)
    download.assert_not_called()
    tensorflow.saved_model.load.assert_called_once_with('/tmp/local-model')
    self.assertIs(result, tensorflow.saved_model.load.return_value)


if __name__ == '__main__':
  absltest.main()
