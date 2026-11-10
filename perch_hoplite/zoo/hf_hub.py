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

# coding=utf-8
"""Wrapper for downloading and staging model files from the Hugging Face Hub.

The Hugging Face sibling of `kaggle_hub.py`. It fetches files or stages remote
model directories to local paths — **format-agnostic**: files may be ONNX,
TFLite, weights, a label list, etc. Source (HF vs Kaggle) is independent of
format, so model classes consume local paths and never embed download or staging
logic. `huggingface_hub` and `etils[epath]` are core dependencies.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import threading

from etils import epath
import huggingface_hub

_STAGE_LOCK = threading.Lock()
_LOCAL_DIR_CACHE: dict[str, str] = {}
_FULLY_STAGED_DIRS: set[str] = set()


def _is_storage_path(model_path: str) -> bool:
  """Returns True if `model_path` is a filesystem or URI path, not a repo ID."""
  return model_path.startswith(('/', '.', '~')) or '://' in model_path


def _copy_dir(src: epath.Path, dst: epath.Path) -> None:
  """Recursively copies a directory from `epath` storage to a local path."""
  dst.mkdir(parents=True, exist_ok=True)
  for child in src.iterdir():
    if child.is_dir():
      _copy_dir(child, dst / child.name)
    else:
      child.copy(dst / child.name)


def _get_or_create_stage_dir(cache_key: str, dir_name: str) -> str:
  """Returns the cached local staging directory for `cache_key`."""
  cached = _LOCAL_DIR_CACHE.get(cache_key)
  if cached is None or not os.path.isdir(cached):
    parent_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, parent_dir, ignore_errors=True)
    cached = os.path.join(parent_dir, dir_name or 'model')
    os.makedirs(cached, exist_ok=True)
    _LOCAL_DIR_CACHE[cache_key] = cached
    _FULLY_STAGED_DIRS.discard(cache_key)
  return cached


def download(repo_id: str, filename: str, local_dir: str | None = None) -> str:
  """Download `filename` from `repo_id`; return the local path.

  With `local_dir`, the file is placed there (out of the shared HF cache);
  otherwise it lands in the default Hugging Face cache.

  Args:
    repo_id: The Hugging Face repo ID.
    filename: The filename to download.
    local_dir: Optional directory to place the file in.

  Returns:
    The local path to the downloaded file.
  """
  if local_dir:
    os.makedirs(local_dir, exist_ok=True)
    return huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir
    )
  return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)


def stage_local_dir(model_path: str) -> str:
  """Stages remote storage directories locally; passes local/Hub IDs through.

  Args:
    model_path: Local directory, `epath` directory URI/path, or Hub repo ID.

  Returns:
    A local POSIX directory path if `model_path` is a filesystem/URI directory,
    or `model_path` unchanged when it is a Hugging Face Hub repo ID.

  Raises:
    FileNotFoundError: If `model_path` looks like a filesystem/URI path rather
      than a Hub repo ID, and the directory does not exist.
  """
  if os.path.isdir(model_path) or not _is_storage_path(model_path):
    return model_path

  src = epath.Path(model_path)
  key = src.as_posix()
  with _STAGE_LOCK:
    cached = _LOCAL_DIR_CACHE.get(key)
    if key in _FULLY_STAGED_DIRS and cached and os.path.isdir(cached):
      return cached
    if not src.is_dir():
      raise FileNotFoundError(f'Model directory not found: {model_path}')
    local_dir = _get_or_create_stage_dir(key, src.name)
    _copy_dir(src, epath.Path(local_dir))
    _FULLY_STAGED_DIRS.add(key)
    return local_dir


def resolve_file(model_path: str, filename: str) -> str:
  """Resolves `filename` from a local/remote directory or Hugging Face repo ID.

  Args:
    model_path: Local directory, `epath` directory URI/path, or Hub repo ID.
    filename: Relative file path within the model directory or Hub repository.

  Returns:
    Local POSIX path to the resolved or downloaded file.

  Raises:
    FileNotFoundError: If `model_path` is a local or `epath` storage path and
      `filename` does not exist within it.
  """
  if os.path.isdir(model_path):
    local_file = os.path.join(model_path, filename)
    if not os.path.isfile(local_file):
      raise FileNotFoundError(f'Model file not found: {local_file}')
    return local_file

  if not _is_storage_path(model_path):
    return download(model_path, filename)

  src_dir = epath.Path(model_path)
  key = src_dir.as_posix()
  with _STAGE_LOCK:
    cached = _LOCAL_DIR_CACHE.get(key)
    if cached and os.path.isdir(cached):
      dst = os.path.join(cached, filename)
      if os.path.isfile(dst):
        return dst
    src_file = src_dir / filename
    if not src_file.exists():
      raise FileNotFoundError(f'Model file not found: {src_file}')
    local_dir = _get_or_create_stage_dir(key, src_dir.name)
    dst = os.path.join(local_dir, filename)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_file.copy(dst)
    return dst
