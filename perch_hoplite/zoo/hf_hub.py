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
"""Wrapper for downloading model files from the Hugging Face Hub.

The Hugging Face sibling of `kaggle_hub.py`. It fetches a file and returns a
local path — **format-agnostic**: the file may be ONNX, TFLite, weights, a label
list, etc. Source (HF vs Kaggle) is independent of format, so model classes
consume a local path and never embed download logic. `huggingface_hub` is a core
dependency, imported normally.
"""

from __future__ import annotations

import os

import huggingface_hub


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
