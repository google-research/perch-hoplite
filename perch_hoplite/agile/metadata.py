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


class AgileMetadata:
  """Handling of agile metadata from CSV files.

  This class reads metadata from CSV files:
  - deployments_metadata.csv: Metadata for each deployment.
  - recordings_metadata.csv: Metadata for each recording.
  - annotations.csv: Annotations for recordings.
  - metadata_description.csv: Description of columns in deployment/recording
    metadata files.
    Expected columns: field_name, metadata_level, type, description.
  """

  def __init__(
      self,
      deployments_path: str | epath.Path,
      recordings_path: str | epath.Path,
      description_path: str | epath.Path,
      annotations_path: str | epath.Path,
  ):
    """Initializes AgileMetadata by reading metadata from CSV files.

    Args:
      deployments_path: Path to deployments_metadata.csv.
      recordings_path: Path to recordings_metadata.csv.
      description_path: Path to metadata_description.csv.
      annotations_path: Path to annotations.csv.
    """
    deployments_path = epath.Path(deployments_path)
    recordings_path = epath.Path(recordings_path)
    description_path = epath.Path(description_path)
    annotations_path = epath.Path(annotations_path)
    self.description_df = pd.DataFrame()
    self.fields = {}
    self.deployment_metadata = {}
    self.recording_metadata = {}
    self.deployment_key_field = None
    self.recording_key_field = None
    self.annotations_df = pd.DataFrame()
    self.annotations = collections.defaultdict(list)

    if annotations_path.exists():
      self.annotations_df = pd.read_csv(
          annotations_path,
          dtype={
              'recording': str,
              'label': str,
              'start_offset_s': float,
              'end_offset_s': float,
              'label_type': str,
          },
      )
      for _, row in self.annotations_df.iterrows():
        label_type = None
        for lt in hoplite_interface.LabelType:
          if row['label_type'].lower() == lt.name.lower():
            label_type = lt
            break
        if label_type is None:
          continue
        self.annotations[row['recording']].append(
            hoplite_interface.Annotation(
                id=-1,
                recording_id=-1,
                label=row['label'],
                offsets=[row['start_offset_s'], row['end_offset_s']],
                label_type=label_type,
                provenance='agile_metadata',
            )
        )

    if not description_path.exists():
      return

    self.description_df = pd.read_csv(description_path)
    for _, row in self.description_df.iterrows():
      self.fields[row.field_name] = MetadataField(
          field_name=row.field_name,
          metadata_level=row.metadata_level,
          dtype=row.type,
          description=row.description if 'description' in row else '',
      )

    deployment_fields = [
        f for f, v in self.fields.items() if v.metadata_level == 'deployment'
    ]
    recording_fields = [
        f for f, v in self.fields.items() if v.metadata_level == 'recording'
    ]

    deployment_df = pd.DataFrame()
    if deployment_fields and deployments_path.exists():
      deployment_df = pd.read_csv(
          deployments_path,
          usecols=deployment_fields,
      )
    recording_df = pd.DataFrame()
    if recording_fields and recordings_path.exists():
      recording_df = pd.read_csv(
          recordings_path,
          usecols=recording_fields,
      )
    self._cast_dtypes(deployment_df, recording_df)

    self.deployment_key_field = None
    if deployment_fields and not deployment_df.empty:
      self.deployment_key_field = deployment_fields[0]
      self.deployment_metadata = {
          r[self.deployment_key_field]: r
          for r in deployment_df.to_dict(orient='records')
      }

    self.recording_key_field = None
    if recording_fields and not recording_df.empty:
      self.recording_key_field = recording_fields[0]
      self.recording_metadata = {
          r[self.recording_key_field]: r
          for r in recording_df.to_dict(orient='records')
      }

  @classmethod
  def from_directory(cls, metadata_dir: str | epath.Path) -> 'AgileMetadata':
    """Creates an AgileMetadata instance from a directory."""
    metadata_dir = epath.Path(metadata_dir)
    return cls(
        metadata_dir / 'deployments_metadata.csv',
        metadata_dir / 'recordings_metadata.csv',
        metadata_dir / 'metadata_description.csv',
        metadata_dir / 'annotations.csv',
    )

  def _cast_dtypes(
      self, deployment_df: pd.DataFrame, recording_df: pd.DataFrame
  ):
    """Casts dataframe columns to specified dtypes."""
    if not deployment_df.empty:
      for field_name in deployment_df.columns:
        field = self.fields[field_name]
        deployment_df[field_name] = deployment_df[field_name].apply(field.cast)
    if not recording_df.empty:
      for field_name in recording_df.columns:
        field = self.fields[field_name]
        recording_df[field_name] = recording_df[field_name].apply(field.cast)

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
      return {k: v for k, v in metadata.items() if not pd.isna(v)}
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
      return {k: v for k, v in metadata.items() if not pd.isna(v)}
    return {}

  def get_recording_annotations(
      self, recording: str
  ) -> list[hoplite_interface.Annotation]:
    """Returns annotations for the recording."""
    return self.annotations[recording]
