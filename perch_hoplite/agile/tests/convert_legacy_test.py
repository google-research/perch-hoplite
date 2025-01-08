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

"""Tests for conversion from previous agile modeling format.

Generating the embeddings requires a call to the Chirp inference library,
which creates a dependency that we don't want to include in the package
dependencies. Instead, we generate the embeddings once and check them into
the repo as a testdata file.
"""

import os
import shutil
import tempfile

from etils import epath
from ml_collections import config_dict
from perch_hoplite import audio_io
from perch_hoplite import path_utils
from perch_hoplite.agile import convert_legacy
from perch_hoplite.agile.tests import test_utils

from absl.testing import absltest
from absl.testing import parameterized


class ConvertLegacyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def write_placeholder_embeddings(
      self, audio_glob, classes, filenames, embed_dir
  ):
    """Method used to generate embeddings for testing."""
    from chirp.inference import embed_lib
    from chirp.inference import tf_examples

    source_infos = embed_lib.create_source_infos([audio_glob], shard_len_s=-1)
    self.assertLen(source_infos, len(classes) * len(filenames))

    # Set up embedding function.
    config = config_dict.ConfigDict()
    model_kwargs = {
        'sample_rate': 16000,
        'embedding_size': 128,
        'make_embeddings': True,
        'make_logits': False,
        'make_separated_audio': False,
        'window_size_s': 5.0,
        'hop_size_s': 5.0,
    }
    embed_fn_config = config_dict.ConfigDict()
    embed_fn_config.write_embeddings = True
    embed_fn_config.write_logits = False
    embed_fn_config.write_separated_audio = False
    embed_fn_config.write_raw_audio = False
    embed_fn_config.write_frontend = False
    embed_fn_config.model_key = 'placeholder_model'
    embed_fn_config.model_config = model_kwargs
    embed_fn_config.min_audio_s = 0.1
    embed_fn_config.file_id_depth = 1
    config.embed_fn_config = embed_fn_config
    config.source_file_patterns = [audio_glob]
    config.num_shards_per_file = -1
    config.shard_len_s = -1

    epath.Path(embed_dir).mkdir(parents=True, exist_ok=True)
    embed_lib.maybe_write_config(config, epath.Path(embed_dir))

    embed_fn = embed_lib.EmbedFn(**embed_fn_config)
    embed_fn.setup()

    # Write embeddings.
    audio_loader = lambda fp, offset: audio_io.load_audio_window(
        fp, offset, sample_rate=16000, window_size_s=120.0
    )
    audio_iterator = audio_io.multi_load_audio_window(
        filepaths=[s.filepath for s in source_infos],
        offsets=[0 for _ in source_infos],
        audio_loader=audio_loader,
    )
    with tf_examples.EmbeddingsTFRecordMultiWriter(
        output_dir=embed_dir, num_files=1
    ) as file_writer:
      for source_info, audio in zip(source_infos, audio_iterator):
        file_id = source_info.file_id(1)
        offset_s = source_info.shard_num * source_info.shard_len_s
        example = embed_fn.audio_to_example(file_id, offset_s, audio)
        file_writer.write(example.SerializeToString())
      file_writer.flush()

  @parameterized.product(
      db_type=(
          'in_mem',
          'sqlite_usearch',
      ),
      generate_embeddings=(False,),
  )
  def test_convert_legacy(self, db_type, generate_embeddings):
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    audio_glob = test_utils.make_wav_files(
        self.tempdir, classes, filenames, file_len_s=60.0
    )
    if generate_embeddings:
      # TODO(tomdenton): Compare the checked-in embeddings to the generated
      # embeddings to ensure that they are the same.
      embed_dir = os.path.join(self.tempdir, 'embeddings')
      self.write_placeholder_embeddings(
          audio_glob, classes, filenames, embed_dir
      )
    else:
      config_path = path_utils.get_absolute_path(
          'agile/tests/testdata/config.json'
      )
      embed_dir = epath.Path(config_path).parent

    if db_type == 'sqlite_usearch':
      kwargs = {'db_path': self.tempdir}
    elif db_type == 'in_mem':
      kwargs = {
          'max_size': 100,
          'degree_bound': 10,
      }
    else:
      raise ValueError(f'Unknown db type: {db_type}')

    db = convert_legacy.convert_tfrecords(
        embeddings_path=embed_dir,
        db_type=db_type,
        dataset_name='test',
        **kwargs,
    )
    # There are six one-minute test files, so we should get 72 embeddings.
    self.assertEqual(db.count_embeddings(), 72)
    metadata = db.get_metadata(key=None)
    self.assertIn('legacy_config', metadata)
    self.assertIn('audio_sources', metadata)
    self.assertIn('model_config', metadata)


if __name__ == '__main__':
  absltest.main()
