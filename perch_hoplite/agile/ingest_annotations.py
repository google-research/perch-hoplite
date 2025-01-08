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

"""Ingest fully-annotated dataset audio and labels."""

import collections
import dataclasses
from typing import Callable

from etils import epath
from ml_collections import config_dict
import pandas as pd
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.db import db_loader
from perch_hoplite.db import interface
from perch_hoplite.taxonomy import annotations_fns
import tqdm

BASE_PATH = epath.Path('gs://chirp-public-bucket/soundscapes/')


@dataclasses.dataclass
class AnnotatedDatasetIngestor:
  """Add annotations to embeddings DB from CSV annotations.

  Note that currently we only add positive labels.

  Attributes:
    base_path: Base path for the dataset.
    audio_glob: Glob for the audio files.
    dataset_name: Name of the dataset.
    annotation_filename: Filename for the annotations CSV.
    annotation_load_fn: Function to load the annotations CSV.
  """

  base_path: epath.Path
  audio_glob: str
  dataset_name: str
  annotation_filename: str
  annotation_load_fn: Callable[[str | epath.Path], pd.DataFrame]

  def ingest_dataset(
      self,
      db: interface.HopliteDBInterface,
      window_size_s: float,
      provenance: str = 'annotations_csv',
  ) -> collections.defaultdict[str, int]:
    """Load annotations and insert labels into the DB.

    Args:
      db: The DB to insert labels into.
      window_size_s: The window size of the embeddings.
      provenance: The provenance to use for the labels.

    Returns:
      A dictionary of ingested label counts.
    """
    annos_path = epath.Path(self.base_path) / self.annotation_filename
    annos_df = self.annotation_load_fn(annos_path)
    lbl_counts = collections.defaultdict(int)
    file_ids = annos_df['filename'].unique()
    label_set = set()
    for file_id in tqdm.tqdm(file_ids):
      embedding_ids = db.get_embeddings_by_source(self.dataset_name, file_id)
      source_annos = annos_df[annos_df['filename'] == file_id]
      for idx in embedding_ids:
        source = db.get_embedding_source(idx)
        window_start = source.offsets[0]
        window_end = window_start + window_size_s
        emb_annos = source_annos[source_annos['start_time_s'] < window_end]
        emb_annos = emb_annos[emb_annos['end_time_s'] > window_start]
        # All of the remianing annotations match the target embedding.
        for labels in emb_annos['label']:
          for label in labels:
            label_set.add(label)
            lbl = interface.Label(
                idx,
                label,
                interface.LabelType.POSITIVE,
                provenance=provenance,
            )
            db.insert_label(lbl)
            lbl_counts[label] += 1
    lbl_count = sum(lbl_counts.values())
    print(f'\nInserted {lbl_count} labels.')
    return lbl_counts


CORNELL_LOADER = lambda x: annotations_fns.load_cornell_annotations(
    x, file_id_prefix='audio/'
)

PRESETS: dict[str, AnnotatedDatasetIngestor] = {
    'powdermill': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'powdermill',
        audio_glob='*/*.wav',
        dataset_name='powdermill',
        annotation_filename='powdermill.csv',
        annotation_load_fn=annotations_fns.load_powdermill_annotations,
    ),
    'hawaii': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'hawaii',
        dataset_name='hawaii',
        audio_glob='audio/*.flac',
        annotation_filename='annotations.csv',
        annotation_load_fn=CORNELL_LOADER,
    ),
    'high_sierras': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'high_sierras',
        dataset_name='high_sierras',
        audio_glob='audio/*.flac',
        annotation_filename='annotations.csv',
        annotation_load_fn=CORNELL_LOADER,
    ),
    'coffee_farms': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'coffee_farms',
        dataset_name='coffee_farms',
        audio_glob='audio/*.flac',
        annotation_filename='annotations.csv',
        annotation_load_fn=CORNELL_LOADER,
    ),
    'peru': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'peru',
        dataset_name='peru',
        audio_glob='audio/*.flac',
        annotation_filename='annotations.csv',
        annotation_load_fn=CORNELL_LOADER,
    ),
    'ssw': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'ssw',
        dataset_name='ssw',
        audio_glob='audio/*.flac',
        annotation_filename='annotations.csv',
        annotation_load_fn=CORNELL_LOADER,
    ),
    'sierras_kahl': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'sierras_kahl',
        dataset_name='sierras_kahl',
        audio_glob='audio/*.flac',
        annotation_filename='annotations.csv',
        annotation_load_fn=CORNELL_LOADER,
    ),
    'anuraset': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'anuraset',
        dataset_name='anuraset',
        audio_glob='raw_data/*/*.wav',
        annotation_filename='annotations.csv',
        annotation_load_fn=annotations_fns.load_anuraset_annotations,
    ),
}


def embed_annotated_dataset(
    ds_choice: str | AnnotatedDatasetIngestor,
    db_path: str,
    db_model_config: embed.ModelConfig,
) -> tuple[interface.HopliteDBInterface, dict[str, int]]:
  """Embed a fully-annotated dataset to SQLite Hoplite DB.

  Args:
    ds_choice: The preset name of the dataset to embed. Alternatively, an
      AnnotatedDatasetIngestor can be provided.
    db_path: The path to the DB.
    db_model_config: The model config for the DB.

  Returns:
    The DB and a dictionary of label counts.
  """
  if isinstance(ds_choice, str):
    ingestor = PRESETS[ds_choice]
  else:
    ingestor = ds_choice

  audio_srcs_config = source_info.AudioSources(
      audio_globs=(
          source_info.AudioSourceConfig(
              dataset_name=ingestor.dataset_name,
              base_path=ingestor.base_path.as_posix(),
              file_glob=ingestor.audio_glob,
              min_audio_len_s=1.0,
              target_sample_rate_hz=-2,
          ),
      )
  )
  db = db_loader.create_new_usearch_db(
      db_path=db_path, embedding_dim=db_model_config.embedding_dim
  )
  print('Initialized DB located at ', db_path)
  worker = embed.EmbedWorker(
      audio_sources=audio_srcs_config, db=db, model_config=db_model_config
  )
  worker.process_all()
  print(f'DB contains {db.count_embeddings()} embeddings.')

  if not hasattr(worker.embedding_model, 'window_size_s'):
    raise ValueError(
        'Model does not have a defined window size, which is needed to compute '
        'relevant annotations for each embedding.'
    )
  window_size_s = getattr(worker.embedding_model, 'window_size_s')
  class_counts = ingestor.ingest_dataset(db, window_size_s=window_size_s)
  db.commit()
  return db, class_counts
