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

"""Iterator utilities."""

from collections.abc import Iterator

import numpy as np


def random_batched_iterator(
    ids: np.ndarray,
    batch_size: int,
    rng: np.random.RandomState,
) -> Iterator[np.ndarray]:
  """Yields batched embedding ids, shuffled after each of unlimited epochs."""
  if batch_size > len(ids):
    raise ValueError('Not enough ids to fill a batch.')
  rng.shuffle(ids)
  q = 0
  while True:
    if q + batch_size > len(ids):
      partial_batch = ids[q : len(ids)].copy()
      overflow = batch_size - len(partial_batch)
      rng.shuffle(ids)
      overflow_batch = ids[:overflow]
      batch = np.concatenate([partial_batch, overflow_batch], axis=0)
      q = overflow
    else:
      batch = ids[q : q + batch_size]
      q += batch_size
    yield batch.copy()
