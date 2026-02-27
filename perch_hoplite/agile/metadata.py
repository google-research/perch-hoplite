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

"""Tooling for handling metadata with Hoplite databases."""

import collections
import dataclasses
from typing import Any

from etils import epath
import pandas as pd
from perch_hoplite.db import interface as hoplite_interface


@dataclasses.dataclass
class MetadataField:
  """Describes a metadata field.

  Attributes:
    field_name: The name of the metadata field.
    metadata_level: The level at which metadata applies, either `deployment` or
      `recording`.
    dtype: The data type of the field. Supported types are `str`, `float`,
      `int`, and `bytes`.
    description: An optional description of the field.
  """

  field_name: str
  metadata_level: str
  dtype: str
  description: str = ''

  def cast(self, value: Any) -> Any:
    """Casts a value to the field's dtype."""
    if pd.isna(value):
      return None
    if self.dtype == 'str':
      return str(value)
    if self.dtype == 'float':
      return float(value)
    if self.dtype == 'int':
      return int(value)
    if self.dtype == 'bytes':
      return str(value).encode('utf-8')
    return value


def _cast_df_dtypes(df: pd.DataFrame, fields: dict[str, MetadataField]):
  """Casts dataframe columns to specified dtypes."""
  if not df.empty:
    for field_name in df.columns:
      if field_name in fields:
        field = fields[field_name]
        df[field_name] = df[field_name].apply(field.cast)


@dataclasses.dataclass
class AgileMetadata:
  """Handling of agile metadata from CSV files.

  Attributes:
    deployment_metadata: Metadata for each deployment.
    recording_metadata: Metadata for each recording.
    annotations: Annotations for recordings.
    fields: Description of columns in deployment/recording metadata files.
  """

  deployment_metadata: dict[str, dict[str, Any]] = dataclasses.field(
      default_factory=dict
  )
  recording_metadata: dict[str, dict[str, Any]] = dataclasses.field(
      default_factory=dict
  )
  annotations: collections.defaultdict[
      str, list[hoplite_interface.Annotation]
  ] = dataclasses.field(default_factory=lambda: collections.defaultdict(list))
  fields: dict[str, MetadataField] = dataclasses.field(default_factory=dict)

  @classmethod
  def from_dataframes(
      cls,
      description_df: pd.DataFrame | None = None,
      deployment_df: pd.DataFrame | None = None,
      recording_df: pd.DataFrame | None = None,
      annotations_df: pd.DataFrame | None = None,
  ) -> 'AgileMetadata':
    """Creates an AgileMetadata instance from pandas dataframes.

    Args:
      description_df: DataFrame with metadata descriptions. Expected columns:
        field_name, metadata_level, type, description.
      deployment_df: DataFrame with deployment metadata.
      recording_df: DataFrame with recording metadata.
      annotations_df: DataFrame with annotations. Expected columns: recording,
        label, start_offset_s, end_offset_s, label_type.

    Returns:
      AgileMetadata instance.
    """
    fields = {}
    deployment_metadata = {}
    recording_metadata = {}
    annotations = collections.defaultdict(list)

    if annotations_df is not None and not annotations_df.empty:
      for _, row in annotations_df.iterrows():
        label_type = None
        for lt in hoplite_interface.LabelType:
          if row['label_type'].lower() == lt.name.lower():
            label_type = lt
            break
        if label_type is None:
          continue
        annotations[row['recording']].append(
            hoplite_interface.Annotation(
                id=-1,
                recording_id=-1,
                label=row['label'],
                offsets=[row['start_offset_s'], row['end_offset_s']],
                label_type=label_type,
                provenance='agile_metadata',
            )
        )

    # If no description, then no metadata to process. Return early.
    if description_df is None or description_df.empty:
      return cls(
          deployment_metadata=deployment_metadata,
          recording_metadata=recording_metadata,
          annotations=annotations,
          fields=fields,
      )

    for _, row in description_df.iterrows():
      fields[row.field_name] = MetadataField(
          field_name=row.field_name,
          metadata_level=row.metadata_level,
          dtype=row.type,
          description=row.description if 'description' in row else '',
      )

    deployment_df = (
        deployment_df if deployment_df is not None else pd.DataFrame()
    )
    recording_df = recording_df if recording_df is not None else pd.DataFrame()

    # Deployment metadata handling.
    if 'deployment' in deployment_df.columns:
      deployment_fields = [
          f for f, v in fields.items() if v.metadata_level == 'deployment'
      ]
      deployment_df = deployment_df[
          [f for f in deployment_fields if f in deployment_df.columns]
      ].copy()
      _cast_df_dtypes(deployment_df, fields)
      deployment_metadata = {
          r['deployment']: r for r in deployment_df.to_dict(orient='records')
      }
    elif not deployment_df.empty:
      raise ValueError(
          'Deployment metadata provided but deployment column missing.'
      )

    # Recording metadata handling.
    if 'recording' in recording_df.columns:
      recording_fields = [
          f for f, v in fields.items() if v.metadata_level == 'recording'
      ]
      recording_df = recording_df[
          [f for f in recording_fields if f in recording_df.columns]
      ].copy()
      _cast_df_dtypes(recording_df, fields)
      recording_metadata = {
          r['recording']: r for r in recording_df.to_dict(orient='records')
      }
    elif not recording_df.empty:
      raise ValueError(
          'Recording metadata provided but recording column missing.'
      )

    return cls(
        deployment_metadata=deployment_metadata,
        recording_metadata=recording_metadata,
        annotations=annotations,
        fields=fields,
    )

  @classmethod
  def from_csv_files(
      cls,
      deployments_path: str | epath.Path,
      recordings_path: str | epath.Path,
      description_path: str | epath.Path,
      annotations_path: str | epath.Path,
  ) -> 'AgileMetadata':
    """Creates an AgileMetadata instance from CSV files."""
    deployments_path = epath.Path(deployments_path)
    recordings_path = epath.Path(recordings_path)
    description_path = epath.Path(description_path)
    annotations_path = epath.Path(annotations_path)

    annotations_df = pd.DataFrame()
    if annotations_path.exists():
      annotations_df = pd.read_csv(
          annotations_path,
          dtype={
              'recording': str,
              'label': str,
              'start_offset_s': float,
              'end_offset_s': float,
              'label_type': str,
          },
      )

    description_df = pd.DataFrame()
    if description_path.exists():
      description_df = pd.read_csv(description_path, index_col=False)

    deployment_df = pd.DataFrame()
    if deployments_path.exists():
      deployment_df = pd.read_csv(deployments_path, index_col=False)
    recording_df = pd.DataFrame()
    if recordings_path.exists():
      recording_df = pd.read_csv(recordings_path, index_col=False)
    return cls.from_dataframes(
        description_df, deployment_df, recording_df, annotations_df
    )

  @classmethod
  def from_directory(cls, metadata_dir: str | epath.Path) -> 'AgileMetadata':
    """Creates an AgileMetadata instance from a directory."""
    metadata_dir = epath.Path(metadata_dir)
    return cls.from_csv_files(
        metadata_dir / 'hoplite_deployments_metadata.csv',
        metadata_dir / 'hoplite_recordings_metadata.csv',
        metadata_dir / 'hoplite_metadata_description.csv',
        metadata_dir / 'hoplite_annotations.csv',
    )

  def get_deployment_metadata(self, deployment: str) -> dict[str, Any]:
    """Returns metadata for the deployment.

    Args:
      deployment: The deployment name to use for matching.

    Returns:
      A dictionary of metadata for the matching deployment, or {}.
      The dictionary will not contain keys where values are None.
    """
    metadata = self.deployment_metadata.get(deployment)
    if metadata:
      return {k: v if not pd.isna(v) else None for k, v in metadata.items()}
    return {}

  def get_recording_metadata(self, recording: str) -> dict[str, Any]:
    """Returns metadata for the recording.

    Args:
      recording: The recording name to use for matching.

    Returns:
      A dictionary of metadata for the matching recording, or {}.
      The dictionary will not contain keys where values are None.
    """
    metadata = self.recording_metadata.get(recording)
    if metadata:
      return {k: v if not pd.isna(v) else None for k, v in metadata.items()}
    return {}

  def get_recording_annotations(
      self, recording: str
  ) -> list[hoplite_interface.Annotation]:
    """Returns annotations for the recording."""
    return self.annotations[recording]
