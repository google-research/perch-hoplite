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

"""Audio IO utilities."""

import concurrent
import itertools
import logging
import os
import tempfile
from typing import Callable, Generator, Sequence
import warnings

from etils import epath
import librosa
import numpy as np
from perch_hoplite import path_utils
import requests
import soundfile


def load_audio(
    path: epath.PathLike,
    target_sample_rate: int,
    dtype: str = 'float32',
    **kwargs,
) -> np.ndarray:
  """Load a general audio resource."""
  path = os.fspath(path)
  if path.startswith('xc'):
    return load_xc_audio(path, target_sample_rate, dtype=dtype)
  elif path.startswith('http'):
    return load_url_audio(path, target_sample_rate, dtype=dtype)
  else:
    return load_audio_file(path, target_sample_rate, dtype=dtype, **kwargs)


def load_audio_file(
    filepath: str | epath.Path,
    target_sample_rate: int,
    resampling_type: str = 'polyphase',
    dtype: str = 'float32',
) -> np.ndarray:
  """Read an audio file, and resample it using librosa."""
  filepath = epath.Path(filepath)
  if target_sample_rate <= 0:
    # Use the native sample rate.
    target_sample_rate = None
  extension = os.path.splitext(filepath)[-1].lower()
  if extension in ('wav', 'flac', 'ogg', 'opus'):
    with filepath.open('rb') as f:
      sf = soundfile.SoundFile(file=f)
      audio = sf.read()
      if target_sample_rate is not None:
        audio = librosa.resample(
            y=audio,
            orig_sr=sf.samplerate,
            target_sr=target_sample_rate,
            res_type=resampling_type,
        )
      return audio.astype(dtype)

  # Handle other audio formats.
  # Because librosa passes file handles to soundfile, we need to copy the file
  # to a temporary file before passing it to librosa.
  with tempfile.NamedTemporaryFile(
      mode='w+b', suffix=extension, delete=False
  ) as f:
    with filepath.open('rb') as sf:
      f.write(sf.read())
  # librosa outputs lots of warnings which we can safely ignore when
  # processing all Xeno-Canto files and PySoundFile is unavailable.
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    audio, _ = librosa.load(
        f.name,
        sr=target_sample_rate,
        res_type=resampling_type,
    )
  os.unlink(f.name)
  return audio.astype(dtype)


def load_audio_window_soundfile(
    filepath: str,
    offset_s: float,
    sample_rate: int,
    window_size_s: float,
    dtype: str = 'float32',
) -> np.ndarray:
  """Load an audio window using Soundfile.

  Args:
    filepath: Path to audio file.
    offset_s: Read offset within the file.
    sample_rate: Sample rate for returned audio.
    window_size_s: Length of audio to read. Reads all if <0.
    dtype: Audio dtype.

  Returns:
    Numpy array of loaded audio.
  """
  with epath.Path(filepath).open('rb') as f:
    sf = soundfile.SoundFile(f)
    if offset_s > 0:
      offset = int(np.float32(offset_s) * sf.samplerate)
      sf.seek(offset)
    if window_size_s < 0:
      a = sf.read()
    else:
      window_size = int(window_size_s * sf.samplerate)
      a = sf.read(window_size)
  if len(a.shape) == 2:
    # Downstream ops expect mono audio, so reduce to mono.
    a = a[:, 0]
  if sample_rate > 0:
    a = librosa.resample(
        y=a, orig_sr=sf.samplerate, target_sr=sample_rate, res_type='polyphase'
    )
  return a.astype(dtype)


def load_audio_window(
    filepath: str,
    offset_s: float,
    sample_rate: int,
    window_size_s: float,
    dtype: str = 'float32',
) -> np.ndarray:
  """Load a slice of audio from a file, hopefully efficiently."""
  # TODO(tomdenton): Find a reliable way to load a flac audio window.
  # If a flac file has the incorrect length in its header, seeking past the
  # end of the file causes the system to hang. This is a bad enough outcome
  # that we don't risk it.
  try:
    return load_audio_window_soundfile(
        filepath, offset_s, sample_rate, window_size_s, dtype
    )
  except soundfile.LibsndfileError:
    logging.info('Failed to load audio with libsndfile: %s', filepath)
  # This fail-over is much slower but more reliable; the entire audio file
  # is loaded (and possibly resampled) and then we extract the target audio.
  audio = load_audio(filepath, sample_rate)
  offset = int(offset_s * sample_rate)
  window_size = int(window_size_s * sample_rate)
  return audio[offset : offset + window_size].astype(dtype)


def multi_load_audio_window(
    filepaths: Sequence[str],
    offsets: Sequence[float] | None,
    audio_loader: Callable[[str, float], np.ndarray],
    max_workers: int = 5,
    buffer_size: int = -1,
) -> Generator[np.ndarray, None, None]:
  """Generator for loading audio windows in parallel.

  Note that audio is returned in the same order as the filepaths.
  Also, this ultimately relies on soundfile, which can be buggy in some cases.

  Caution: Because this generator uses an Executor, it can continue holding
  resources while not being used. If you are using this in a notebook, you
  should use this in a 'nameless' context, like:
  ```
  for audio in multi_load_audio_window(...):
    ...
  ```
  or in a try/finally block:
  ```
  audio_iterator = multi_load_audio_window(...)
  try:
    for audio in audio_iterator:
      ...
  finally:
    del(audio_iterator)
  ```
  Otherwise, the generator will continue to hold resources until the notebook
  is closed.

  Args:
    filepaths: Paths to audio to load.
    offsets: Read offset in seconds for each file, or None if no offsets are
      needed.
    audio_loader: Function to load audio given a filepath and offset.
    max_workers: Number of threads to allocate.
    buffer_size: Max number of audio windows to queue up. Defaults to 10x the
      number of workers.

  Yields:
    Loaded audio windows.
  """
  if buffer_size == -1:
    buffer_size = 10 * max_workers
  if offsets is None:
    offsets = [0.0 for _ in filepaths]

  # TODO(tomdenton): Use itertools.batched in Python 3.12+
  def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
      yield batch

  task_iterator = zip(filepaths, offsets)
  batched_iterator = batched(task_iterator, buffer_size)
  mapping = lambda x: audio_loader(x[0], x[1])

  executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
  try:
    yield from itertools.chain.from_iterable(
        executor.map(mapping, batch) for batch in batched_iterator
    )
  finally:
    executor.shutdown(wait=False, cancel_futures=True)


def load_xc_audio(
    xc_id: str, sample_rate: int, dtype: str = 'float32'
) -> np.ndarray:
  """Load audio from Xeno-Canto given an ID like 'xc12345'."""
  if not xc_id.startswith('xc'):
    raise ValueError(f'XenoCanto id {xc_id} does not start with "xc".')
  xc_id = xc_id[2:]
  try:
    int(xc_id)
  except ValueError as exc:
    raise ValueError(f'XenoCanto id xc{xc_id} is not an integer.') from exc
  session = requests.Session()
  session.mount(
      'https://',
      requests.adapters.HTTPAdapter(
          max_retries=requests.adapters.Retry(total=5, backoff_factor=0.1)
      ),
  )
  url = f'https://xeno-canto.org/{xc_id}/download'
  try:
    data = session.get(url=url).content
  except requests.exceptions.RequestException as e:
    raise requests.exceptions.RequestException(
        f'Failed to load audio from Xeno-Canto {xc_id}'
    ) from e
  with tempfile.NamedTemporaryFile(suffix='.mp3', mode='wb', delete=False) as f:
    f.write(data)
    f.flush()
  audio = load_audio_file(f.name, target_sample_rate=sample_rate)
  os.unlink(f.name)
  return audio.astype(dtype)


def load_url_audio(
    url: str, sample_rate: int, dtype: str = 'float32'
) -> np.ndarray:
  """Load audio from a URL."""
  data = requests.get(url).content
  with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
    f.write(data)
    f.flush()
  audio = load_audio_file(f.name, target_sample_rate=sample_rate)
  os.unlink(f.name)
  return audio.astype(dtype)
