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

"""Functionality for embedding audio examples."""

from concurrent import futures
import dataclasses
import itertools
import threading

from absl import logging
import audioread
from ml_collections import config_dict
import numpy as np
from perch_hoplite import audio_io
from perch_hoplite.agile import source_info
from perch_hoplite.db import interface as hoplite_interface
from perch_hoplite.zoo import model_configs
from perch_hoplite.zoo import zoo_interface
import soundfile


@dataclasses.dataclass
class ModelConfig(hoplite_interface.HopliteConfig):
  """Configuration for embedding model.

  Attributes:
    model_key: Key for the model wrapper class.
    embedding_dim: Dimensionality of the embedding.
    model_config: Config dict of arguments to instantiate the model wrapper.
    logit_key: If provided, model predictions will be stored instead of raw
      embeddings.
    logit_idxes: When storing model predictions, allows selecting a subset of
      prediction classes.
  """

  model_key: str
  embedding_dim: int
  model_config: config_dict.ConfigDict
  logits_key: str | None = None
  logits_idxes: tuple[int, ...] | None = None


def worker_initializer(state):
  name = threading.current_thread().name
  state[name + 'db'] = state['db'].thread_split()


def process_source_id(
    state,
    source_id: source_info.SourceId,
    window_size_s: float,
):
  """Process a single audio source."""
  worker = state['worker']
  name = threading.current_thread().name
  db = state[name + 'db']
  glob = worker.audio_globs[source_id.dataset_name]
  target_sample_rate = worker.get_sample_rate_hz(source_id)
  audio_array = worker.load_audio(source_id)

  if audio_array is None:
    return
  if (
      audio_array.shape[0]
      < glob.min_audio_len_s * worker.embedding_model.sample_rate
  ):
    return

  embs = db.match_window_ids(
      deployments_filter=config_dict.create(
          eq=dict(project=source_id.dataset_name)
      ),
      recordings_filter=config_dict.create(eq=dict(filename=source_id.file_id)),
      windows_filter=config_dict.create(
          approx=dict(
              offsets=[source_id.offset_s, source_id.offset_s + window_size_s],
          )
      ),
  )
  if embs:
    return

  outputs = worker.embedding_model.embed(audio_array)
  logits_key = state['worker'].model_config.logits_key
  if logits_key is None:
    embeddings = outputs.embeddings
  else:
    embeddings = outputs.logits[logits_key]
    logits_idxes = state['worker'].model_config.logits_idxes
    if logits_idxes is not None:
      embeddings = embeddings[..., logits_idxes]
    # Add channel axis to match the expected shape of the embeddings.
    embeddings = embeddings[:, np.newaxis, :]

  if embeddings is None:
    return

  sources = []
  offsets = []
  embs = []

  hop_size_s = worker.compute_hop_size_s(source_id, target_sample_rate)
  for t, embedding in enumerate(embeddings):
    offset_s = source_id.offset_s + t * hop_size_s
    offsets_list = [offset_s, offset_s + window_size_s]
    for channel_embedding in embedding:
      sources.append(source_id)
      offsets.append(offsets_list)
      embs.append(channel_embedding)
  return sources, offsets, embs


# TODO(tomdenton): Use itertools.batched in Python 3.12+
def batched(iterable, n):
  it = iter(iterable)
  while batch := tuple(itertools.islice(it, n)):
    yield batch


class EmbedWorker:
  """Worker for embedding audio examples."""

  def __init__(
      self,
      audio_sources: source_info.AudioSources,
      model_config: ModelConfig,
      db: hoplite_interface.HopliteDBInterface,
      embedding_model: zoo_interface.EmbeddingModel | None = None,
      audio_worker_threads: int = 8,
  ):
    self.db = db
    self.model_config = model_config
    self.audio_sources = audio_sources
    self.audio_worker_threads = audio_worker_threads
    if embedding_model is None:
      model_class = model_configs.get_model_class(model_config.model_key)
      self.embedding_model = model_class.from_config(model_config.model_config)
    else:
      self.embedding_model = embedding_model
    self.window_size_s = getattr(self.embedding_model, 'window_size_s')
    self.audio_globs = {
        g.dataset_name: g for g in self.audio_sources.audio_globs
    }

  def _log_error(self, source_id, exception, counter_name):
    logging.warning(
        'The audio at (%s / %f) could not be loaded (%s). '
        'The exception was (%s)',
        source_id.filepath,
        source_id.offset_s,
        counter_name,
        exception,
    )

  def _update_audio_sources(self):
    """Validates the embed config and/or saves it to the DB."""
    db_metadata = self.db.get_metadata(None)
    if 'audio_sources' not in db_metadata:
      self.db.insert_metadata(
          'audio_sources', self.audio_sources.to_config_dict()
      )
      return

    db_audio_sources = source_info.AudioSources.from_config_dict(
        db_metadata['audio_sources']
    )
    merged = self.audio_sources.merge_update(db_audio_sources)
    self.db.insert_metadata('audio_sources', merged.to_config_dict())
    self.audio_sources = merged

  def _update_model_config(self):
    """Validates the model config and/or saves it to the DB."""
    db_metadata = self.db.get_metadata(None)
    if 'model_config' not in db_metadata:
      self.db.insert_metadata(
          'model_config', self.model_config.to_config_dict()
      )
      return

    db_model_config = ModelConfig(**db_metadata['model_config'])
    if self.model_config == db_model_config:
      return

    # Validate the config against the DB.
    # TODO(tomdenton): Implement compatibility checks for model configs.
    if self.model_config.model_key != db_model_config.model_key:
      raise AssertionError(
          'The configured model key does not match the model key that is '
          'already in the DB.'
      )
    if self.model_config.embedding_dim != db_model_config.embedding_dim:
      raise AssertionError(
          'The configured embedding dimension does not match the embedding '
          'dimension that is already in the DB.'
      )
    self.db.insert_metadata('model_config', self.model_config.to_config_dict())

  def update_configs(self):
    """Validates the configs and saves them to the DB."""
    self._update_model_config()
    self._update_audio_sources()
    self.db.commit()

  def get_sample_rate_hz(self, source_id: source_info.SourceId) -> int:
    """Get the sample rate of the embedding model."""
    dataset_name = source_id.dataset_name
    if dataset_name not in self.audio_globs:
      raise ValueError(f'Dataset name {dataset_name} not found in audio globs.')
    audio_glob = self.audio_globs[dataset_name]
    if audio_glob.target_sample_rate_hz == -2:
      return self.embedding_model.sample_rate
    elif audio_glob.target_sample_rate_hz == -1:
      # Uses the file's native sample rate.
      return -1
    elif audio_glob.target_sample_rate_hz > 0:
      return audio_glob.target_sample_rate_hz
    else:
      raise ValueError('Invalid target_sample_rate.')

  def load_audio(self, source_id: source_info.SourceId) -> np.ndarray | None:
    """Load audio from the indicated source and log any problems."""
    target_sample_rate_hz = self.get_sample_rate_hz(source_id)
    try:
      audio_array = audio_io.load_audio_window(
          filepath=source_id.filepath,
          offset_s=source_id.offset_s,
          sample_rate=target_sample_rate_hz,
          window_size_s=source_id.shard_len_s,
      )
      return np.array(audio_array)
    except soundfile.LibsndfileError as inst:
      self._log_error(source_id, inst, 'audio_libsndfile_error')
    except ValueError as inst:
      self._log_error(source_id, inst, 'audio_bad_offset')
    except audioread.NoBackendError as inst:
      self._log_error(source_id, inst, 'audio_no_backend')
    except EOFError as inst:
      self._log_error(source_id, inst, 'audio_eof_error')
    except RuntimeError as inst:
      if 'Soundfile is not available' in str(inst):
        self._log_error(source_id, inst, 'audio_no_soundfile')
      else:
        self._log_error(source_id, inst, 'audio_runtime_error')

  def compute_hop_size_s(
      self,
      source_id: source_info.SourceId,
      target_sample_rate_hz: int,
      model_hop_size_s: float | None = None,
  ) -> float:
    """Compute the hop size of the embedding model."""

    if model_hop_size_s is not None:
      return model_hop_size_s
    if hasattr(self.embedding_model, 'hop_size_s'):
      model_hop_size_s = float(self.embedding_model.hop_size_s)
    else:
      # TODO(tomdenton): Allow user specified hop size.
      raise ValueError('hop_size_s is not defined for the model.')
    model_sample_rate = self.embedding_model.sample_rate

    if target_sample_rate_hz == -2:
      return model_hop_size_s
    elif target_sample_rate_hz == -1:
      audio_sample_rate = source_id.sample_rate_hz
    elif target_sample_rate_hz > 0:
      audio_sample_rate = target_sample_rate_hz
    else:
      raise ValueError('Invalid target_sample_rate.')
    return model_hop_size_s * model_sample_rate / audio_sample_rate

  def embedding_exists(
      self,
      source_id: source_info.SourceId,
      window_size_s: float,
  ) -> bool:
    """Check whether embeddings already exist for the given source ID."""
    embs = self.db.match_window_ids(
        deployments_filter=config_dict.create(
            eq=dict(project=source_id.dataset_name)
        ),
        recordings_filter=config_dict.create(
            eq=dict(filename=source_id.file_id)
        ),
        windows_filter=config_dict.create(
            approx=dict(
                offsets=[
                    source_id.offset_s,
                    source_id.offset_s + window_size_s,
                ],
            )
        ),
        limit=1,
    )
    return bool(embs)

  def process_all(self, target_dataset_name: str | None = None, batch_size=32):
    """Process all audio examples."""

    # Update model config and audio sources in the database.
    self.update_configs()

    # Create missing deployments and recordings in the database.
    source_id_to_deployment_id = {}
    source_id_to_recording_id = {}
    for source in self.audio_sources.iterate_all_sources(target_dataset_name):
      source_id = source.to_id()

      deployments = self.db.get_all_deployments(
          config_dict.create(eq=dict(project=source.dataset_name))
      )
      if not deployments:
        deployment_id = self.db.insert_deployment(
            name=source.dataset_name, project=source.dataset_name
        )
      else:
        deployment_id = deployments[0].id
      source_id_to_deployment_id[source_id] = deployment_id

      recordings = self.db.get_all_recordings(
          config_dict.create(
              eq=dict(
                  filename=source.file_id,
                  deployment_id=deployment_id,
              )
          )
      )
      if not recordings:
        recording_id = self.db.insert_recording(
            filename=source.file_id, deployment_id=deployment_id
        )
      else:
        recording_id = recordings[0].id
      source_id_to_recording_id[source_id] = recording_id

    # Commit all changes for deployments and recordings to the database.
    self.db.commit()

    # Process all sources.
    state = {}
    state['db'] = self.db
    state['worker'] = self
    with futures.ThreadPoolExecutor(
        max_workers=self.audio_worker_threads,
        initializer=worker_initializer,
        initargs=(state,),
    ) as executor:
      source_iterator = self.audio_sources.iterate_all_sources(
          target_dataset_name
      )
      for source_ids_batch in batched(source_iterator, batch_size):
        got = executor.map(
            process_source_id,
            itertools.repeat(state),
            source_ids_batch,
            itertools.repeat(self.window_size_s),
        )
        # TODO(tomdenton): Consider using a db writer thread to avoid blocking.
        for result in got:
          if result is None:
            continue
          for source, offsets, embedding in zip(*result):
            source_id = source.to_id()
            recording_id = source_id_to_recording_id[source_id]
            self.db.insert_window(
                recording_id=recording_id,
                embedding=embedding,
                offsets=offsets,
            )

    # Commit all changes for windows to the database.
    self.db.commit()
