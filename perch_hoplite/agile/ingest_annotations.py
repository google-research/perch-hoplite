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

"""Ingest fully-annotated dataset audio and labels."""

import collections
from collections.abc import Callable
import dataclasses

from etils import epath
from ml_collections import config_dict
import pandas as pd
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.db import db_loader
from perch_hoplite.db import interface
from perch_hoplite.taxonomy import annotations_fns
import tqdm

BASE_PATH = epath.Path('gs://chirp-public-bucket/soundscapes')


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
      provenance: str = 'annotations_csv',
  ) -> collections.defaultdict[str, int]:
    """Load annotations and insert labels into the DB.

    Args:
      db: The DB to insert labels into.
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
      source_annos = annos_df[annos_df['filename'] == file_id]
      for idx in db.match_window_ids(
          deployments_filter=config_dict.create(
              eq=dict(project=self.dataset_name)
          ),
          recordings_filter=config_dict.create(eq=dict(filename=file_id)),
      ):
        window = db.get_window(idx)
        start_offset = window.offsets[0]
        end_offset = window.offsets[1]
        emb_annos = source_annos[source_annos['start_time_s'] < end_offset]
        emb_annos = emb_annos[emb_annos['end_time_s'] > start_offset]
        # All of the remaining annotations match the target embedding.
        for labels in emb_annos['label']:
          for label in labels:
            label_set.add(label)
            db.insert_annotation(
                recording_id=window.recording_id,
                offsets=window.offsets,
                label=label,
                label_type=interface.LabelType.POSITIVE,
                provenance=provenance,
            )
            lbl_counts[label] += 1
    lbl_count = sum(lbl_counts.values())
    print(f'\nInserted {lbl_count} labels.')
    return lbl_counts


CORNELL_LOADER = lambda x: annotations_fns.load_cornell_annotations(
    x, file_id_prefix='audio/'
)


PRESETS: dict[str, AnnotatedDatasetIngestor] = {
    'anuraset': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'anuraset',
        dataset_name='anuraset',
        audio_glob='raw_data/*/*.wav',
        annotation_filename='annotations.csv',
        annotation_load_fn=annotations_fns.load_anuraset_annotations,
    ),
    'coffee_farms': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'coffee_farms',
        dataset_name='coffee_farms',
        audio_glob='audio/*.flac',
        annotation_filename='annotations.csv',
        annotation_load_fn=CORNELL_LOADER,
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
    'peru': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'peru',
        dataset_name='peru',
        audio_glob='audio/*.flac',
        annotation_filename='annotations.csv',
        annotation_load_fn=CORNELL_LOADER,
    ),
    'powdermill': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'powdermill',
        audio_glob='*/*.wav',
        dataset_name='powdermill',
        annotation_filename='powdermill.csv',
        annotation_load_fn=annotations_fns.load_powdermill_annotations,
    ),
    'sierras_kahl': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'sierras_kahl',
        dataset_name='sierras_kahl',
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
    'weldy_calltype': AnnotatedDatasetIngestor(
        base_path=BASE_PATH / 'weldy_calltype',
        dataset_name='weldy_calltype',
        audio_glob='annotated_recordings/*.wav',
        annotation_filename='annotations.csv',
        annotation_load_fn=annotations_fns.load_weldy_annotations,
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

  class_counts = ingestor.ingest_dataset(db)
  db.commit()
  return db, class_counts
