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

"""Tools for producing hoplite metadata files from various sources."""

import glob
import os

from etils import epath
import pandas as pd


def convert_wabad_data(
    annotations_csv_path: str, metadata_csv_path: str, output_dir: str
):
  """Converts WABAD pooled annotations to hoplite metadata files.

  Args:
    annotations_csv_path: Path to the WABAD annotations CSV file.
    metadata_csv_path: Path to the WABAD metadata CSV file.
    output_dir: Directory to write hoplite_* files to.
  """
  output_dir = epath.Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f'Reading WABAD annotations data from {annotations_csv_path}...')
  # Read CSV with comma separator and dot decimal.
  df = pd.read_csv(annotations_csv_path, sep=',', decimal='.')

  # Create the recording identifier by combining site and filename.
  # This assumes audio files will be accessible like:
  # 'ARD/Recordings/ARD_20211027_072000.wav'
  df['recording_id'] = df['Site'] + '/Recordings/' + df['Recording']

  # 1. Create hoplite_annotations.csv
  print('Generating hoplite_annotations.csv...')
  annotations_df = pd.DataFrame({
      'recording': df['recording_id'],
      'label': df['Species'],
      'start_offset_s': df['Begin_Time_(s)'],
      'end_offset_s': df['End_Time_(s)'],
      'label_type': 'positive',
  })
  annotations_df.to_csv(output_dir / 'hoplite_annotations.csv', index=False)

  # 2. Create hoplite_deployments_metadata.csv
  print(f'Reading WABAD metadata from {metadata_csv_path}...')
  meta_df = pd.read_csv(metadata_csv_path, encoding='latin-1')
  print('Generating hoplite_deployments_metadata.csv...')
  deployment_df = meta_df[[
      'Site ID',
      'Recording location',
      'Biome',
      'Latitude',
      'Longitude',
  ]].copy()
  deployment_df = deployment_df.rename(
      columns={
          'Site ID': 'deployment',
          'Recording location': 'country',
          'Biome': 'biome',
          'Latitude': 'lat',
          'Longitude': 'lon',
      }
  )
  deployment_df = deployment_df.drop_duplicates(
      subset=['deployment']
  ).reset_index(drop=True)
  deployment_df.to_csv(
      output_dir / 'hoplite_deployments_metadata.csv', index=False
  )

  # 3. Create hoplite_metadata_description.csv
  print('Generating hoplite_metadata_description.csv...')
  description_content = (
      'field_name,metadata_level,type,description\n'
      'deployment,deployment,str,"WABAD recording site identifier"\n'
      'country,deployment,str,"Country of recording site"\n'
      'biome,deployment,str,"Biome of recording site"\n'
      'lat,deployment,float,"Latitude of recording site"\n'
      'lon,deployment,float,"Longitude of recording site"\n'
  )
  (output_dir / 'hoplite_metadata_description.csv').write_text(
      description_content
  )

  print(f'\nConversion complete. Files written to {output_dir}')
  print(f'Found {len(annotations_df)} annotations.')
  print(f'Found {len(deployment_df)} unique deployments.')


def convert_anuraset_labels(strong_labels_dir: str, output_csv_path: str):
  """Converts Anuraset strong labels to hoplite_annotations.csv.

  Searches for txt files in strong_labels_dir/*/*.txt, parses them as TSV
  with columns start_s, end_s, label, and converts them to hoplite
  annotations CSV format.

  Args:
    strong_labels_dir: Directory containing the Anuraset strong labels.
    output_csv_path: Path to write the hoplite_annotations.csv file to.
  """
  all_annotations = []
  txt_files = glob.glob(os.path.join(strong_labels_dir, '*/*.txt'))
  print(f'Found {len(txt_files)} label files.')

  def clean_label(label):
    """Removes trailing _[A-Z] from label if present."""
    if (
        isinstance(label, str)
        and len(label) > 2
        and label[-2] == '_'
        and label[-1].isalpha()
        and label[-1].isupper()
    ):
      return label[:-2]
    return label

  for txt_file in txt_files:
    try:
      # Use usecols=[0, 1, 2] to only read the first three columns,
      # ignoring any extra columns caused by extra tabs in a row.
      df = pd.read_csv(
          txt_file,
          sep='\t',
          header=None,
          names=['start_s', 'end_s', 'label'],
          usecols=[0, 1, 2],
      )
    except pd.errors.EmptyDataError:
      print(f'Skipping empty file: {txt_file}')
      continue
    except Exception as e:  # pylint: disable=broad-except
      print(f'Could not parse file {txt_file}: {e}')
      continue

    relative_path = os.path.relpath(txt_file, strong_labels_dir)
    recording_name = os.path.splitext(relative_path)[0] + '.wav'

    for _, row in df.iterrows():
      all_annotations.append({
          'recording': recording_name,
          'label': clean_label(row['label']),
          'start_offset_s': row['start_s'],
          'end_offset_s': row['end_s'],
          'label_type': 'positive',
      })

  if not all_annotations:
    print('No annotations found.')
    return

  annotations_df = pd.DataFrame(all_annotations)
  # Ensure columns are in the right order.
  annotations_df = annotations_df[[
      'recording',
      'label',
      'start_offset_s',
      'end_offset_s',
      'label_type',
  ]]
  annotations_df.to_csv(output_csv_path, index=False)
  print(f'{len(annotations_df)} annotations saved to {output_csv_path}')
