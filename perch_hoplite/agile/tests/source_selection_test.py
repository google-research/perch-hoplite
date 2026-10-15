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

"""Tests for recursive and explicit audio source selection."""

from pathlib import Path
from unittest import mock
import tempfile

from absl.testing import absltest
from perch_hoplite.agile import source_info


class AudioSourceSelectionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = Path(self.tmp.name)
    for filename in (
        'top.wav',
        'deployment/audio.wav',
        'deployment/nested/child.wav',
        'deployment/nested/skip.txt',
        'another/file.wav',
    ):
      path = self.root / filename
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes(b'fixture')
    self.length = mock.patch.object(
        source_info.audio_io,
        'get_file_length_s_and_sample_rate',
        return_value=(4.0, 16000),
    )
    self.length.start()
    self.addCleanup(self.length.stop)

  def sources(self, config):
    return source_info.AudioSources(audio_globs=(config,))

  def make_config(self, **kwargs):
    return source_info.AudioSourceConfig(
        dataset_name='test',
        base_path=str(self.root),
        **kwargs,
    )

  def test_default_glob_is_non_recursive(self):
    globs = self.sources(self.make_config(file_glob='*.wav'))
    found = list(globs.iterate_all_sources())
    self.assertEqual([s.file_id for s in found], ['top.wav'])

  def test_recursive_glob_includes_nested_audio_files(self):
    globs = self.sources(self.make_config(file_glob='*.wav', recursive=True))
    found = list(globs.iterate_all_sources())
    self.assertEqual(
        [s.file_id for s in found],
        [
            'another/file.wav',
            'deployment/audio.wav',
            'deployment/nested/child.wav',
            'top.wav',
        ],
    )
    self.assertEqual([s.dataset_name for s in found], ['test'] * 4)

  def test_double_star_glob_is_supported_only_with_recursion(self):
    with self.assertRaises(NotImplementedError):
      list(
          self.sources(
              self.make_config(file_glob='**/*.wav')
          ).iterate_all_sources()
      )
    globs = self.sources(self.make_config(file_glob='**/*.wav', recursive=True))
    self.assertLen(list(globs.iterate_all_sources()), 4)

  def test_recursive_subfolder_pattern_is_not_a_global_basename_match(self):
    globs = self.sources(
        self.make_config(file_glob='deployment/*.wav', recursive=True)
    )
    found = list(globs.iterate_all_sources())
    self.assertEqual([s.file_id for s in found], ['deployment/audio.wav'])

  def test_explicit_list_preserves_requested_order(self):
    globs = self.sources(
        self.make_config(
            file_paths=(
                'deployment/nested/child.wav',
                'top.wav',
            ),
        )
    )
    found = list(globs.iterate_all_sources())
    self.assertEqual(
        [s.file_id for s in found],
        ['deployment/nested/child.wav', 'top.wav'],
    )
    self.assertTrue(all(s.filepath.startswith(str(self.root)) for s in found))

  def test_absolute_paths_inside_base_path_work(self):
    globs = self.sources(
        self.make_config(
            file_paths=(str(self.root / 'another/file.wav'),),
        )
    )
    found = list(globs.iterate_all_sources())
    self.assertEqual([s.file_id for s in found], ['another/file.wav'])

  def test_repeated_explicit_paths_are_deduplicated(self):
    globs = self.sources(
        self.make_config(
            file_paths=('top.wav', 'top.wav', 'deployment/audio.wav'),
        )
    )
    self.assertEqual(
        [s.file_id for s in globs.iterate_all_sources()],
        ['top.wav', 'deployment/audio.wav'],
    )

  def test_explicit_files_support_existing_shard_limits(self):
    globs = self.sources(
        self.make_config(
            file_paths=('top.wav',),
            shard_len_s=1.0,
            max_shards_per_file=2,
        )
    )
    found = list(globs.iterate_all_sources())
    self.assertEqual([s.offset_s for s in found], [0, 1.0])
    self.assertTrue(all(s.file_id == 'top.wav' for s in found))

  def test_empty_explicit_list_yields_no_sources(self):
    globs = self.sources(self.make_config(file_paths=()))
    self.assertEmpty(list(globs.iterate_all_sources()))

  def test_missing_explicit_file_fails(self):
    globs = self.sources(self.make_config(file_paths=('not-a-file.wav',)))
    with self.assertRaises(FileNotFoundError):
      list(globs.iterate_all_sources())

  def test_file_path_escape_fails(self):
    for path in ('../escape.wav', '/outside/project.wav'):
      with self.subTest(path=path):
        globs = self.sources(self.make_config(file_paths=(path,)))
        with self.assertRaises(ValueError):
          list(globs.iterate_all_sources())

  def test_conflicting_or_missing_selection_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      self.make_config()
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      self.make_config(file_glob='*.wav', file_paths=('top.wav',))
    with self.assertRaisesRegex(ValueError, 'only valid'):
      self.make_config(file_paths=('top.wav',), recursive=True)
    with self.assertRaises(TypeError):
      self.make_config(file_paths='top.wav')

  def test_source_config_serialization_roundtrips_new_options(self):
    original = self.sources(
        self.make_config(
            file_paths=('top.wav', 'deployment/audio.wav'),
        )
    )
    encoded = original.to_config_dict()
    restored = source_info.AudioSources.from_config_dict(encoded)
    self.assertEqual(
        [s.file_id for s in restored.iterate_all_sources()],
        ['top.wav', 'deployment/audio.wav'],
    )

  def test_recursive_config_serialization_roundtrip(self):
    original = self.sources(self.make_config(file_glob='*.wav', recursive=True))
    restored = source_info.AudioSources.from_config_dict(
        original.to_config_dict()
    )
    self.assertLen(list(restored.iterate_all_sources()), 4)

  def test_dataset_filtering_with_mixed_file_selectors(self):
    sources = source_info.AudioSources(
        audio_globs=(
            self.make_config(file_glob='*.wav'),
            source_info.AudioSourceConfig(
                dataset_name='other',
                base_path=str(self.root),
                file_paths=('deployment/audio.wav',),
            ),
        )
    )
    found = list(sources.iterate_all_sources('other'))
    self.assertEqual([s.file_id for s in found], ['deployment/audio.wav'])
    self.assertEqual(found[0].dataset_name, 'other')


if __name__ == '__main__':
  absltest.main()
