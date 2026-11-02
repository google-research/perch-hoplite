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

"""Functionality for embedding audio examples."""

from collections.abc import Sequence
from concurrent import futures
import dataclasses
import datetime
import itertools
import threading
from typing import Literal

from absl import logging
import audioread
from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite import audio_io
from perch_hoplite.agile import metadata
from perch_hoplite.agile import source_info
from perch_hoplite.agile import timestamp_resolver as ts_resolver
from perch_hoplite.db import datatypes
from perch_hoplite.db import interface as hoplite_interface
from perch_hoplite.taxonomy import namespace
from perch_hoplite.taxonomy import namespace_db
from perch_hoplite.zoo import model_configs
from perch_hoplite.zoo import zoo_interface
import soundfile
import tqdm


@dataclasses.dataclass
class ModelConfig(datatypes.HopliteConfig):
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


def _get_model_class_list(
    embedding_model: zoo_interface.EmbeddingModel,
    logits_key: str | None = None,
) -> tuple[str | None, namespace.ClassList]:
  """Retrieve (logits_key, ClassList) from an embedding model."""
  model_cl = getattr(embedding_model, 'class_list', None)
  if model_cl is None and hasattr(embedding_model, 'embedding_model'):
    model_cl = getattr(embedding_model.embedding_model, 'class_list', None)
  if isinstance(model_cl, namespace.ClassList):
    return logits_key, model_cl
  if isinstance(model_cl, dict) and model_cl:
    if logits_key in model_cl:
      return logits_key, model_cl[logits_key]
    if 'label' in model_cl:
      return 'label', model_cl['label']
    first_key = next(iter(model_cl))
    return first_key, model_cl[first_key]
  raise ValueError('Embedding model does not provide a class_list.')


def _resolve_detection_classes(
    embedding_model: zoo_interface.EmbeddingModel,
    class_list: namespace_db.ClassListType | Sequence[str] | None,
    logits_key: str | None = None,
) -> tuple[str | None, tuple[int, ...], tuple[str, ...]]:
  """Resolve target detection classes and their indices in the model logits."""
  resolved_key, model_cl = _get_model_class_list(embedding_model, logits_key)
  if class_list is None:
    target_classes = model_cl.classes
  elif isinstance(class_list, (str, namespace.ClassList)):
    target_classes = namespace_db.get_classes(class_list)
  else:
    target_classes = tuple(class_list)

  class_to_idx = {c: i for i, c in enumerate(model_cl.classes)}
  missing = [c for c in target_classes if c not in class_to_idx]
  if missing:
    raise ValueError(f'Classes not found in model class_list: {missing}')
  target_idxes = tuple(class_to_idx[c] for c in target_classes)
  return resolved_key, target_idxes, target_classes


def extract_detections(
    logits: np.ndarray,
    target_idxes: tuple[int, ...],
    target_classes: tuple[str, ...],
    min_score: float,
    base_offset_s: float,
    hop_size_s: float,
    window_size_s: float,
) -> list[tuple[list[float], str, float]]:
  """Extract and merge adjacent detections above min_score."""
  scores = logits[..., target_idxes]
  if scores.ndim == 3:
    scores = np.max(scores, axis=1)

  detected_classes = np.where(np.any(scores > min_score, axis=0))[0]
  detections = []
  for cls_idx in detected_classes:
    label = target_classes[cls_idx]
    frame_idxes = np.where(scores[:, cls_idx] > min_score)[0]
    curr_start = base_offset_s + float(frame_idxes[0]) * hop_size_s
    curr_end = curr_start + window_size_s
    curr_score = float(scores[frame_idxes[0], cls_idx])
    for t in frame_idxes[1:]:
      start_s = base_offset_s + float(t) * hop_size_s
      end_s = start_s + window_size_s
      score = float(scores[t, cls_idx])
      if start_s <= curr_end + 1e-5:
        curr_end = max(curr_end, end_s)
        curr_score = max(curr_score, score)
      else:
        detections.append(([curr_start, curr_end], label, curr_score))
        curr_start, curr_end, curr_score = start_s, end_s, score
    detections.append(([curr_start, curr_end], label, curr_score))
  return detections


def worker_initializer(state):
  name = threading.current_thread().name
  state[name + 'db'] = state['db'].thread_split()


def process_source_id(
    state,
    source_id: source_info.SourceId,
    window_size_s: float,
    recording_timestamp: datetime.datetime | None = None,
):
  """Process a single audio source."""
  worker = state['worker']
  # To access the thread-specific DB, we need to use the thread name.
  # name = threading.current_thread().name
  # db = state[name + 'db']
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

  outputs = worker.embedding_model.embed(audio_array)
  hop_size_s = worker.compute_hop_size_s(source_id, target_sample_rate)
  detections = worker.get_detections(outputs, source_id.offset_s, hop_size_s)
  if not worker.write_embeddings:
    return source_id, [], [], [], detections

  logits_key = worker.model_config.logits_key
  if logits_key is None:
    embeddings = outputs.embeddings
  else:
    embeddings = outputs.logits[logits_key]
    logits_idxes = worker.model_config.logits_idxes
    if logits_idxes is not None:
      embeddings = embeddings[..., logits_idxes]
    # Add channel axis to match the expected shape of the embeddings.
    embeddings = embeddings[:, np.newaxis, :]

  if embeddings is None:
    return

  offsets = []
  embs = []
  timestamps = []

  for t, embedding in enumerate(embeddings):
    offset_s = source_id.offset_s + t * hop_size_s
    offsets_list = [offset_s, offset_s + window_size_s]
    ts = worker.timestamp_resolver.get_filepath_timestamp(
        source_id.filepath, offset_s, base_timestamp=recording_timestamp
    )
    for channel_embedding in embedding:
      offsets.append(offsets_list)
      embs.append(channel_embedding)
      timestamps.append(ts)

  return source_id, offsets, embs, timestamps, detections


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
      timestamp_resolver: ts_resolver.TimestampResolver | None = None,
      timestamp_file_pattern: str | None = None,
      write_embeddings: bool = True,
      class_list: namespace_db.ClassListType | Sequence[str] | None = None,
      min_detection_score: float | None = None,
  ):
    self.db = db
    self.model_config = model_config
    self.audio_sources = audio_sources
    self.audio_worker_threads = audio_worker_threads
    self.timestamp_file_pattern = timestamp_file_pattern
    self.write_embeddings = write_embeddings
    self.min_detection_score = min_detection_score
    if timestamp_resolver is None:
      if self.timestamp_file_pattern is not None:
        self.timestamp_resolver = ts_resolver.TimestampFromFilename(
            db=self.db,
            datetime_format=self.timestamp_file_pattern,
        )
      else:
        self.timestamp_resolver = ts_resolver.NoneResolver(db=self.db)
    else:
      self.timestamp_resolver = timestamp_resolver
    if embedding_model is None:
      model_class = model_configs.get_model_class(model_config.model_key)
      self.embedding_model = model_class.from_config(model_config.model_config)
    else:
      self.embedding_model = embedding_model
    if self.min_detection_score is not None:
      (
          self._det_logits_key,
          self._det_target_idxes,
          self._det_target_classes,
      ) = _resolve_detection_classes(
          self.embedding_model, class_list, self.model_config.logits_key
      )
    self.window_size_s = getattr(self.embedding_model, 'window_size_s')
    self.audio_globs = {
        g.dataset_name: g for g in self.audio_sources.audio_globs
    }
    self.metadata = {
        g.dataset_name: metadata.AgileMetadata.from_directory(g.base_path)
        for g in self.audio_sources.audio_globs
    }
    self._deployment_map = {}
    self._recording_map = {}

  def get_detections(
      self,
      outputs: zoo_interface.InferenceOutputs,
      base_offset_s: float,
      hop_size_s: float,
  ) -> list[tuple[list[float], str, float]]:
    """Extract merged detections from model outputs."""
    if self.min_detection_score is None or not outputs.logits:
      return []
    if self._det_logits_key in outputs.logits:
      logits = outputs.logits[self._det_logits_key]
    elif 'label' in outputs.logits:
      logits = outputs.logits['label']
    else:
      logits = next(iter(outputs.logits.values()))
    return extract_detections(
        logits=logits,
        target_idxes=self._det_target_idxes,
        target_classes=self._det_target_classes,
        min_score=self.min_detection_score,
        base_offset_s=base_offset_s,
        hop_size_s=hop_size_s,
        window_size_s=self.window_size_s,
    )

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

  def _get_or_insert_deployment_id(
      self,
      deployment_name: str,
      project_name: str,
      error_on_insert: bool = False,
  ) -> int:
    """Get the deployment ID for the given deployment name and project name."""
    if (deployment_name, project_name) in self._deployment_map:
      return self._deployment_map[(deployment_name, project_name)]

    deployments = self.db.get_all_deployments(
        config_dict.create(eq=dict(name=deployment_name, project=project_name))
    )
    if deployments:
      deployment_id = deployments[0].id
      self._deployment_map[(deployment_name, project_name)] = deployment_id
      return deployment_id
    if error_on_insert:
      raise ValueError(
          f'Deployment {deployment_name} not found in project {project_name}.'
      )
    md = self.metadata[project_name].get_deployment_metadata(deployment_name)
    md.pop('deployment', None)
    deployment_id = self.db.insert_deployment(
        name=deployment_name,
        project=project_name,
        **md,
    )
    self._deployment_map[(deployment_name, project_name)] = deployment_id
    return deployment_id

  def _get_deployment_name(self, file_id: str, dataset_name: str) -> str:
    """Gets deployment name from recording metadata or file_id."""
    if dataset_name in self.metadata:
      rec_md = self.metadata[dataset_name].get_recording_metadata(file_id)
      if rec_md.get('deployment_name'):
        return str(rec_md['deployment_name'])
    if '/' in file_id:
      return file_id.split('/')[0]
    return dataset_name

  def get_recording_timestamp(
      self, filename: str, dataset_name: str
  ) -> datetime.datetime | None:
    """Gets recording timestamp from metadata or filename pattern."""
    md = self.metadata[dataset_name].get_recording_metadata(filename)
    rec_datetime = md.get('datetime', None)
    if isinstance(rec_datetime, str):
      try:
        rec_datetime = datetime.datetime.fromisoformat(rec_datetime)
      except ValueError:
        pass
    if rec_datetime is None and self.timestamp_file_pattern is not None:
      try:
        rec_datetime = datetime.datetime.strptime(
            epath.Path(filename).stem, self.timestamp_file_pattern
        )
        rec_datetime = rec_datetime.replace(tzinfo=datetime.timezone.utc)
      except ValueError:
        pass
    return rec_datetime  # pyrefly: ignore[bad-return]

  def _get_or_insert_recording_id(
      self,
      filename: str,
      deployment_id: int,
      dataset_name: str,
      error_on_insert: bool = False,
  ) -> tuple[int, bool]:
    """Get the recording ID, and indicate whether it was newly inserted."""
    if (deployment_id, filename) in self._recording_map:
      return self._recording_map[(deployment_id, filename)], False

    recordings = self.db.get_all_recordings(
        config_dict.create(
            eq=dict(filename=filename, deployment_id=deployment_id)
        )
    )
    if recordings:
      recording_id = recordings[0].id
      self._recording_map[(deployment_id, filename)] = recording_id
      return recording_id, False
    if error_on_insert:
      raise ValueError(
          f'Recording {filename} not found in deployment {deployment_id}.'
      )
    md = self.metadata[dataset_name].get_recording_metadata(filename)
    md.pop('recording', None)
    md.pop('datetime', None)
    md.pop('deployment_name', None)
    rec_datetime = self.get_recording_timestamp(filename, dataset_name)
    recording_id = self.db.insert_recording(
        filename=filename,
        datetime=rec_datetime,
        deployment_id=deployment_id,
        **md,
    )
    self._recording_map[(deployment_id, filename)] = recording_id
    return recording_id, True

  def add_deployments(
      self,
      target_dataset_name: str | None = None,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
  ):
    """Add deployments to db and create a source ID to deployment ID mapping."""
    if handle_duplicates != 'allow':
      existing = self.db.get_all_deployments()
      if not existing:
        handle_duplicates = 'allow'
      else:
        for d in existing:
          self._deployment_map[(d.name, d.project)] = d.id

    # Gather unique deployments from sources.
    unique_deployments = set()
    for source in self.audio_sources.iterate_all_sources(target_dataset_name):
      depl_name = self._get_deployment_name(source.file_id, source.dataset_name)
      unique_deployments.add((depl_name, source.dataset_name))

    # Create missing deployments in the database.
    for deployment_name, project_name in unique_deployments:
      if (
          handle_duplicates == 'allow'
          or (deployment_name, project_name) not in self._deployment_map
      ):
        self._get_or_insert_deployment_id(
            deployment_name=deployment_name,
            project_name=project_name,
        )
      elif handle_duplicates == 'error':
        raise ValueError(
            f'Deployment {deployment_name} in project {project_name} '
            'already exists.'
        )
      elif handle_duplicates == 'overwrite':
        self.db.remove_deployment(
            self._deployment_map[(deployment_name, project_name)]
        )
        del self._deployment_map[(deployment_name, project_name)]
        self._get_or_insert_deployment_id(
            deployment_name=deployment_name,
            project_name=project_name,
        )
      elif handle_duplicates == 'skip':
        continue

    self.db.commit()

  def add_recordings(
      self,
      target_dataset_name: str | None = None,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
  ) -> set[int]:
    """Add recordings to db and create a source ID to recording ID mapping."""
    if handle_duplicates != 'allow':
      existing = self.db.get_all_recordings()
      if not existing:
        handle_duplicates = 'allow'
      else:
        for r in existing:
          self._recording_map[(r.deployment_id, r.filename)] = r.id

    new_recordings = set([])
    for source in self.audio_sources.iterate_all_sources(target_dataset_name):
      depl_name = self._get_deployment_name(source.file_id, source.dataset_name)
      deployment_id = self._get_or_insert_deployment_id(
          deployment_name=depl_name,
          project_name=source.dataset_name,
          error_on_insert=True,
      )
      if (
          handle_duplicates == 'allow'
          or (deployment_id, source.file_id) not in self._recording_map
      ):
        recording_id, is_new = self._get_or_insert_recording_id(
            source.file_id,
            deployment_id,
            source.dataset_name,
        )
        if is_new:
          new_recordings.add(recording_id)
      elif handle_duplicates == 'skip':
        continue
      elif handle_duplicates == 'error':
        raise ValueError(
            f'Recording {source.file_id} already exists in deployment '
            f'{deployment_id}.'
        )
      elif handle_duplicates == 'overwrite':
        self.db.remove_recording(
            self._recording_map[(deployment_id, source.file_id)]
        )
        del self._recording_map[(deployment_id, source.file_id)]
        recording_id, _ = self._get_or_insert_recording_id(
            source.file_id,
            deployment_id,
            source.dataset_name,
        )
        new_recordings.add(recording_id)
    self.db.commit()
    return new_recordings

  def add_annotations(
      self,
      target_dataset_name: str | None = None,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
  ):
    """Add annotations from metadata to db."""
    dataset_names = self.metadata.keys()
    if target_dataset_name is not None:
      dataset_names = [target_dataset_name]

    for dataset_name in dataset_names:
      if dataset_name not in self.metadata:
        continue
      agile_md = self.metadata[dataset_name]
      if not agile_md.annotations:
        continue
      for file_id, annotation_list in tqdm.tqdm(
          agile_md.annotations.items(), desc='Adding annotations'
      ):
        depl_name = self._get_deployment_name(file_id, dataset_name)
        if not depl_name:
          logging.warning(
              'Could not get deployment name from file_id %s, skipping.',
              file_id,
          )
          continue
        depl_id = self._get_or_insert_deployment_id(depl_name, dataset_name)
        rec_id, _ = self._get_or_insert_recording_id(
            file_id, depl_id, dataset_name
        )
        for annotation in annotation_list:
          self.db.insert_annotation(
              rec_id,
              annotation.offsets,
              annotation.label,
              annotation.label_type,
              provenance=annotation.provenance,
              handle_duplicates=handle_duplicates,
          )
    self.db.commit()

  def embed_dataset(
      self,
      batch_size=32,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
      target_dataset_name: str | None = None,
      new_recordings: set[int] | None = None,
      commit_every_n_batches: int | None = 100,
  ):
    """Embed audio examples from the given dataset."""
    if self.write_embeddings and self.timestamp_resolver is not None:
      if 'timestamp' not in self.db.get_extra_table_columns().get(
          'windows', {}
      ):
        self.db.add_extra_table_column('windows', 'timestamp', str)
    if self.min_detection_score is not None:
      if 'model_score' not in self.db.get_extra_table_columns().get(
          'annotations', {}
      ):
        self.db.add_extra_table_column('annotations', 'model_score', float)
    # Process all sources.
    state = {}
    state['db'] = self.db
    state['worker'] = self
    state['new_recordings'] = new_recordings
    with futures.ThreadPoolExecutor(
        max_workers=self.audio_worker_threads,
        initializer=worker_initializer,
        initargs=(state,),
    ) as executor:
      source_iterator = self.audio_sources.iterate_all_sources(
          target_dataset_name
      )
      for i, source_ids_batch in enumerate(
          batched(source_iterator, batch_size), start=1
      ):
        recording_timestamps = [
            self.get_recording_timestamp(s.file_id, s.dataset_name)
            for s in source_ids_batch
        ]
        got = executor.map(
            process_source_id,
            itertools.repeat(state),
            source_ids_batch,
            itertools.repeat(self.window_size_s),
            recording_timestamps,
        )
        # TODO(tomdenton): Consider using a db writer thread to avoid blocking.
        for result in got:
          if result is None:
            continue
          source_id, offsets_list, embs_list, timestamps_list, detections = (
              result
          )
          depl_name = self._get_deployment_name(
              source_id.file_id, source_id.dataset_name
          )
          deployment_id = self._get_or_insert_deployment_id(
              depl_name, source_id.dataset_name
          )
          recording_id, _ = self._get_or_insert_recording_id(
              source_id.file_id, deployment_id, source_id.dataset_name
          )
          if new_recordings is not None and recording_id in new_recordings:
            dupe_strategy = 'allow'
          else:
            dupe_strategy = handle_duplicates

          if self.write_embeddings and offsets_list:
            windows_batch = []
            for o, ts in zip(offsets_list, timestamps_list):
              win_dict = {'recording_id': recording_id, 'offsets': o}
              if ts is not None:
                win_dict['timestamp'] = ts
              windows_batch.append(win_dict)
            self.db.insert_windows_batch(
                windows_batch,
                np.array(embs_list),
                handle_duplicates=dupe_strategy,
            )

          for det_offsets, det_label, det_score in detections:
            self.db.insert_annotation(
                recording_id=recording_id,
                offsets=det_offsets,
                label=det_label,
                label_type=datatypes.LabelType.POSITIVE,
                provenance=self.model_config.model_key,
                handle_duplicates=dupe_strategy,
                model_score=det_score,
            )
        if commit_every_n_batches and i % commit_every_n_batches == 0:
          self.db.commit()
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

  def process_all(
      self,
      target_dataset_name: str | None = None,
      batch_size=32,
      handle_duplicates='error',
      commit_every_n_batches: int | None = 128,
  ):
    """Process all audio examples."""

    # Update model config and audio sources in the database.
    self.update_configs()
    if (
        self.db.count_embeddings() == 0
        and not self.db.get_all_deployments()
        and not self.db.get_all_recordings()
    ):
      # No chance of duplicates, so we can use "allow" mode.
      handle_duplicates = 'allow'

    # Add deployments and recordings to the database.
    print('\nAdding deployments...')
    self.add_deployments(target_dataset_name, handle_duplicates)  # pyrefly: ignore[bad-argument-type]
    print('\nAdding recordings...')
    new_recordings = self.add_recordings(target_dataset_name, handle_duplicates)  # pyrefly: ignore[bad-argument-type]
    print('\nAdding annotations...')
    self.add_annotations(handle_duplicates=handle_duplicates)  # pyrefly: ignore[bad-argument-type]
    print('\nEmbedding audio...')
    self.embed_dataset(
        batch_size=batch_size,
        handle_duplicates=handle_duplicates,  # pyrefly: ignore[bad-argument-type]
        target_dataset_name=target_dataset_name,
        new_recordings=new_recordings,
        commit_every_n_batches=commit_every_n_batches,
    )
    self.db.commit()
