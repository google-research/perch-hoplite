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

"""Tests for timestamp_resolver."""

import datetime
import os
import shutil
import tempfile

from absl.testing import absltest
from etils import epath
from ml_collections import config_dict
from perch_hoplite.agile import timestamp_resolver
from perch_hoplite.db import datatypes
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import sqlite_usearch_impl

from absl.testing import parameterized


class ExampleTimestampResolver(timestamp_resolver.TimestampResolver):
  """Concrete class for testing TimestampResolver base behavior."""

  def get_filepath_timestamp(
      self,
      filepath: str | epath.Path,
      subchunk_offset_s: float | None = None,
      base_timestamp: datetime.datetime | None = None,
  ) -> datetime.datetime:
    base_time = base_timestamp or datetime.datetime(
        2023, 1, 1, tzinfo=datetime.timezone.utc
    )
    if subchunk_offset_s is not None:
      return base_time + datetime.timedelta(seconds=subchunk_offset_s)
    return base_time


class TimestampResolverTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def get_db(self, db_type: str):
    if db_type == 'in_mem':
      return in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=32)
    elif db_type == 'sqlite_usearch':
      db_path = os.path.join(self.tempdir, 'test_db')
      usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(32)
      return sqlite_usearch_impl.SQLiteUSearchDB.create(
          db_path=db_path, usearch_cfg=usearch_cfg
      )
    raise ValueError(f'Unknown db_type: {db_type}')

  @parameterized.parameters('in_mem', 'sqlite_usearch')
  def test_get_recording_filepath_simple(self, db_type):
    db = self.get_db(db_type)
    # Setup resolver
    resolver = ExampleTimestampResolver(db=db)

    # Test file that does not exist but starts with '/'
    recording = datatypes.Recording(
        id=1, filename='/abs/path/file.wav', deployment_id=1
    )
    filepath = resolver.get_recording_filepath(recording)
    self.assertEqual(filepath, epath.Path('/abs/path/file.wav'))

  @parameterized.parameters('in_mem', 'sqlite_usearch')
  def test_get_recording_filepath_base_path(self, db_type):
    db = self.get_db(db_type)
    # Setup resolver with base path
    resolver = ExampleTimestampResolver(db=db, base_path='/base/dir')

    recording = datatypes.Recording(id=1, filename='file.wav', deployment_id=1)
    filepath = resolver.get_recording_filepath(recording)
    self.assertEqual(filepath, epath.Path('/base/dir/file.wav'))

  @parameterized.parameters('in_mem', 'sqlite_usearch')
  def test_get_recording_filepath_audio_sources_metadata(self, db_type):
    db = self.get_db(db_type)
    # Store audio_sources metadata in DB
    embed_config = config_dict.ConfigDict(
        {
            'audio_globs': [
                {'base_path': 'gs://metadata/base'},
            ]
        }
    )
    db.insert_metadata('audio_sources', embed_config)

    resolver = ExampleTimestampResolver(db=db)
    recording = datatypes.Recording(id=1, filename='song.wav', deployment_id=1)
    filepath = resolver.get_recording_filepath(recording)
    self.assertEqual(filepath, epath.Path('gs://metadata/base/song.wav'))

  @parameterized.parameters('in_mem', 'sqlite_usearch')
  def test_get_offset_timestamp(self, db_type):
    db = self.get_db(db_type)
    resolver = ExampleTimestampResolver(db=db)
    # Insert recording in DB
    dep_id = db.insert_deployment(name='dep', project='proj')
    rec_id = db.insert_recording(
        filename='/path/file.wav', deployment_id=dep_id
    )

    offset_s = 60.5
    ts = resolver.get_offset_timestamp(rec_id, offset_s)
    self.assertEqual(
        ts,
        datetime.datetime(
            2023, 1, 1, 0, 1, 0, 500000, tzinfo=datetime.timezone.utc
        ),
    )

  @parameterized.parameters('in_mem', 'sqlite_usearch')
  def test_get_window_timestamp(self, db_type):
    db = self.get_db(db_type)
    resolver = ExampleTimestampResolver(db=db)
    # Insert recording and window in DB
    dep_id = db.insert_deployment(name='dep', project='proj')
    rec_id = db.insert_recording(
        filename='/path/file.wav', deployment_id=dep_id
    )
    win_id = db.insert_window(recording_id=rec_id, offsets=[10.0, 15.0])

    ts = resolver.get_window_timestamp(win_id)
    self.assertEqual(
        ts,
        datetime.datetime(2023, 1, 1, 0, 0, 10, tzinfo=datetime.timezone.utc),
    )

  @parameterized.parameters('in_mem', 'sqlite_usearch')
  def test_timestamp_from_filename(self, db_type):
    db = self.get_db(db_type)
    resolver = timestamp_resolver.TimestampFromFilename(
        db=db,
        datetime_format='%Y%m%d_%H%M%S',
        datetime_timezone=datetime.timezone.utc,
    )
    file_path = '/recording/20231024_153000.wav'

    # Without subchunk offset
    ts_no_offset = resolver.get_filepath_timestamp(file_path)
    expected_no_offset = datetime.datetime(
        2023, 10, 24, 15, 30, 0, tzinfo=datetime.timezone.utc
    )
    self.assertEqual(ts_no_offset, expected_no_offset)

    # With subchunk offset
    ts_with_offset = resolver.get_filepath_timestamp(
        file_path, subchunk_offset_s=120.5
    )
    expected_with_offset = datetime.datetime(
        2023, 10, 24, 15, 32, 0, 500000, tzinfo=datetime.timezone.utc
    )
    self.assertEqual(ts_with_offset, expected_with_offset)

    # Test via get_offset_timestamp from database
    dep_id = db.insert_deployment(name='dep', project='proj')
    rec_id = db.insert_recording(
        filename='20231024_153000.wav', deployment_id=dep_id
    )
    ts_from_db = resolver.get_offset_timestamp(rec_id, window_offset_s=120.5)
    self.assertEqual(ts_from_db, expected_with_offset)


if __name__ == '__main__':
  absltest.main()
