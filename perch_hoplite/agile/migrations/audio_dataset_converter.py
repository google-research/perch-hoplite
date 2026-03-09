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

"""Tool for reorganizing audio files and annotations for hoplite ingestion."""

import dataclasses
from etils import epath
from perch_hoplite.agile import embed
from perch_hoplite.agile import metadata
from perch_hoplite.agile import source_info
from perch_hoplite.db import db_loader
from perch_hoplite.zoo import model_configs


@dataclasses.dataclass(kw_only=True)
class AudioDatasetIngestor:
  """Ingests premade audio datasets to the hoplite standard format."""

  dataset_name: str
  audio_globs: list[str]
  annotation_filename: str | epath.Path | None = None

  def load_metadata(self, base_path: epath.Path) -> metadata.AgileMetadata:
    """Load metadata and annotations for the dataset."""
    return metadata.AgileMetadata.from_directory(base_path)

  def execute(
      self,
      base_path: epath.Path | str,
      output_path: epath.Path | str | None,
      model_key: str,
      logits_key: str | None = None,
      logits_idxes: tuple[int, ...] | None = None,
      embedding_dim: int | None = None,
  ):
    """Ingest the dataset into a hoplite DB."""
    preset_info = model_configs.get_preset_model_config(model_key)
    db_model_config = embed.ModelConfig(
        model_key=preset_info.model_key,
        embedding_dim=preset_info.embedding_dim,
        model_config=preset_info.model_config,
        logits_key=logits_key,
        logits_idxes=logits_idxes,
    )

    if isinstance(base_path, str):
      base_path = epath.Path(base_path)
    if isinstance(output_path, str):
      output_path = epath.Path(output_path)

    self.load_metadata(base_path)

    if output_path is None:
      output_path = base_path

    audio_srcs_config = source_info.AudioSources(
        audio_globs=tuple([
            source_info.AudioSourceConfig(
                dataset_name=self.dataset_name,
                base_path=base_path.as_posix(),
                file_glob=glob,
                min_audio_len_s=1.0,
                target_sample_rate_hz=-2,
            )
            for glob in self.audio_globs
        ])
    )
    if embedding_dim is None and logits_idxes is not None:
      embedding_dim = len(logits_idxes)
    elif embedding_dim is None and logits_key is not None:
      raise ValueError(
          'Cannot infer embedding dimension from logits key, please specify'
          ' embedding_dim.'
      )
    elif embedding_dim is None:
      embedding_dim = preset_info.embedding_dim
    db = db_loader.create_new_usearch_db(
        db_path=output_path, embedding_dim=embedding_dim
    )
    print('Initialized DB located at ', output_path)
    worker = embed.EmbedWorker(
        audio_sources=audio_srcs_config, db=db, model_config=db_model_config
    )
    worker.process_all()
    print(f'DB contains {db.count_embeddings()} embeddings.')
    return db
