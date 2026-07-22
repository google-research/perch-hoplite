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

"""Base interfaces for resolving timestamps of audio recordings."""

import abc
import datetime
from absl import logging
from etils import epath
from perch_hoplite.db import datatypes
from perch_hoplite.db import interface


class TimestampResolver(abc.ABC):
  """Base class for resolving database timestamps for a recording or window."""

  def __init__(
      self,
      db: interface.HopliteDBInterface,
      base_path: str | epath.Path | None = None,
  ):
    self.db = db
    self.base_path = epath.Path(base_path) if base_path else None

  def get_recording_filepath(
      self, recording: datatypes.Recording
  ) -> epath.Path:
    """Determines the full filepath for a recording."""

    filename = recording.filename
    path = epath.Path(filename)
    if (
        str(filename).startswith(('gs://', 'http://', 'https://', '/'))
        or path.exists()
    ):
      return path

    if self.base_path:
      candidate = self.base_path / filename
      if (
          str(self.base_path).startswith(('gs://', 'http://', 'https://', '/'))
          or candidate.exists()
      ):
        return candidate

    # Attempt to locate base_path from database metadata
    try:
      embed_config = self.db.get_metadata('audio_sources')
      if embed_config and 'audio_globs' in embed_config:
        for glob_cfg in embed_config['audio_globs']:
          base = glob_cfg.get('base_path')
          if base:
            candidate = epath.Path(base) / filename
            if (
                str(base).startswith(('gs://', 'http://', 'https://'))
                or candidate.exists()
            ):
              return candidate
    except Exception as e:  # pylint: disable=broad-except
      logging.debug('Could not retrieve audio_sources from db metadata: %s', e)

    return path

  def get_filepath_timestamp(
      self,
      filepath: str | epath.Path | None = None,
      subchunk_offset_s: float | None = None,
      base_timestamp: datetime.datetime | None = None,
  ) -> datetime.datetime | None:
    """Computes the exact UTC timestamp for a given offset in a file.

    Args:
      filepath: The filepath to the audio file.
      subchunk_offset_s: The offset in seconds to compute the timestamp for, or
        None if the timestamp for the start of the file should be returned. This
        is necessary for files that are subchunked.
      base_timestamp: The base datetime timestamp of the recording, if already
        known/populated.

    Returns:
      The computed datetime timestamp, or None.
    """
    del filepath
    if base_timestamp is not None:
      if subchunk_offset_s is not None:
        return base_timestamp + datetime.timedelta(seconds=subchunk_offset_s)
      return base_timestamp
    raise NotImplementedError()

  def get_offset_timestamp(
      self, recording_id: int, window_offset_s: float
  ) -> datetime.datetime | None:
    """Computes the exact UTC timestamp for a given offset in a recording."""
    recording = self.db.get_recording(recording_id)
    rec_dt = recording.datetime
    if isinstance(rec_dt, str):
      rec_dt = datetime.datetime.fromisoformat(rec_dt)
    return self.get_filepath_timestamp(
        recording.filename, window_offset_s, base_timestamp=rec_dt
    )

  def get_window_timestamp(self, window_id: int) -> datetime.datetime | None:
    """Computes the exact UTC timestamp for an embedding window ID."""
    window = self.db.get_window(window_id)
    return self.get_offset_timestamp(window.recording_id, window.offsets[0])


class TimestampFromFilename(TimestampResolver):
  """Base class for resolving timestamps directly contained in filenames.

  Attributes:
    db: The Hoplite database interface.
    datetime_format: The format string for parsing the timestamp from the
      filename.
    datetime_timezone: The timezone to use for the timestamp.
  """

  def __init__(
      self,
      db: interface.HopliteDBInterface,
      datetime_format: str | None = None,
      datetime_timezone: datetime.timezone = datetime.timezone.utc,
  ):
    super().__init__(db)
    self.datetime_format = datetime_format
    self.datetime_timezone = datetime_timezone

  def get_filepath_timestamp(
      self,
      filepath: str | epath.Path | None = None,
      subchunk_offset_s: float | None = None,
      base_timestamp: datetime.datetime | None = None,
  ) -> datetime.datetime | None:
    """Computes the exact UTC timestamp for a given offset in a file.

    Assumes that the filename is the string representation of the timestamp.

    Args:
      filepath: The path to the audio file.
      subchunk_offset_s: The offset in seconds to compute the timestamp for, or
        None if the timestamp for the start of the file should be returned.
      base_timestamp: The base datetime timestamp of the recording, if already
        known/populated.

    Returns:
      The computed datetime timestamp, or None.
    """
    try:
      if (
          ts := super().get_filepath_timestamp(
              filepath, subchunk_offset_s, base_timestamp
          )
      ) is not None:
        return ts
    except NotImplementedError:
      pass

    if filepath is None:
      raise ValueError(
          'filepath must be provided if base_timestamp is not set.'
      )
    filepath = epath.Path(filepath)
    timestamp = datetime.datetime.strptime(filepath.stem, self.datetime_format)
    timestamp = timestamp.replace(tzinfo=self.datetime_timezone)
    if subchunk_offset_s:
      timestamp += datetime.timedelta(seconds=subchunk_offset_s)
    return timestamp


class NoneResolver(TimestampResolver):
  """Returns None for all timestamps."""

  def get_filepath_timestamp(
      self,
      filepath: str | epath.Path | None = None,
      subchunk_offset_s: float | None = None,
      base_timestamp: datetime.datetime | None = None,
  ) -> datetime.datetime | None:
    try:
      return super().get_filepath_timestamp(
          filepath, subchunk_offset_s, base_timestamp
      )
    except NotImplementedError:
      return None
