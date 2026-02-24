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

"""Tests for metadata handling."""

import os

from absl.testing import absltest
from perch_hoplite.agile import metadata
from perch_hoplite.db import interface as hoplite_interface


class MetadataTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    self.desc_path = os.path.join(self.temp_dir, 'metadata_description.csv')
    self.dep_path = os.path.join(self.temp_dir, 'deployments_metadata.csv')
    self.rec_path = os.path.join(self.temp_dir, 'recordings_metadata.csv')

    with open(self.desc_path, 'w') as f:
      f.write(
          'field_name,metadata_level,type,description\n'
          'deployment,deployment,str,Deployment name\n'
          'habitat,deployment,str,Habitat type\n'
          'lon,deployment,float,"Longitude, degrees"\n'
          'lat,deployment,float,"Latitude, degrees"\n'
          'recorder_id,deployment,bytes,ID of hardware\n'
          'recording,recording,str,Recording name\n'
          'observer,recording,str,Observer name\n'
          'temp_c,recording,float,"Temperature, C"\n'
      )
    with open(self.dep_path, 'w') as f:
      f.write(
          'deployment,habitat,lon,lat,recorder_id\n'
          'dep_a,"forest",-120.1,30.2,rec_01\n'
          'dep_b,"prairie",-120.2,30.2,rec_02\n'
      )
    with open(self.rec_path, 'w') as f:
      f.write('recording,observer,temp_c\nrec_a,Buffy,,\nrec_b,Willow,22.2\n')

  def test_metadata_loading(self):
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    self.assertLen(md.deployment_metadata, 2)
    self.assertLen(md.recording_metadata, 2)

  def test_get_deployment_metadata(self):
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    dep_a_md = md.get_deployment_metadata('dep_a')
    self.assertEqual(
        dep_a_md,
        {
            'deployment': 'dep_a',
            'habitat': 'forest',
            'lon': -120.1,
            'lat': 30.2,
            'recorder_id': b'rec_01',
        },
    )

  def test_get_recording_metadata(self):
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    rec_b_md = md.get_recording_metadata('rec_b')
    self.assertEqual(
        rec_b_md,
        {'recording': 'rec_b', 'observer': 'Willow', 'temp_c': 22.2},
    )

  def test_get_recording_metadata_missing_value(self):
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    rec_a_md = md.get_recording_metadata('rec_a')
    self.assertEqual(
        rec_a_md,
        {'recording': 'rec_a', 'observer': 'Buffy'},
    )

  def test_get_unknown_metadata(self):
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    self.assertEqual({}, md.get_deployment_metadata('unknown'))
    self.assertEqual({}, md.get_recording_metadata('unknown'))

  def test_missing_description_file(self):
    os.remove(self.desc_path)
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    self.assertEqual({}, md.get_deployment_metadata('dep_a'))
    self.assertEqual({}, md.get_recording_metadata('rec_a'))

  def test_missing_deployment_file(self):
    os.remove(self.dep_path)
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    self.assertEqual({}, md.get_deployment_metadata('dep_a'))
    self.assertLen(md.recording_metadata, 2)

  def test_missing_recording_file(self):
    os.remove(self.rec_path)
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    self.assertLen(md.deployment_metadata, 2)
    self.assertEqual({}, md.get_recording_metadata('rec_a'))

  def test_get_annotations(self):
    ann_path = os.path.join(self.temp_dir, 'annotations.csv')
    with open(ann_path, 'w') as f:
      f.write(
          'recording,label,start_offset_s,end_offset_s,label_type\n'
          'rec_a,species_a,0.5,1.5,positive\n'
          'rec_a,species_b,2.0,3.0,positive\n'
          'rec_b,species_a,1.0,2.0,negative\n'
      )
    md = metadata.AgileMetadata.from_directory(self.temp_dir)
    rec_a_ann = md.get_recording_annotations('rec_a')
    self.assertEqual(
        rec_a_ann,
        [
            hoplite_interface.Annotation(
                id=-1,
                recording_id=-1,
                label='species_a',
                offsets=[0.5, 1.5],
                label_type=hoplite_interface.LabelType.POSITIVE,
                provenance='agile_metadata',
            ),
            hoplite_interface.Annotation(
                id=-1,
                recording_id=-1,
                label='species_b',
                offsets=[2.0, 3.0],
                label_type=hoplite_interface.LabelType.POSITIVE,
                provenance='agile_metadata',
            ),
        ],
    )
    rec_b_ann = md.get_recording_annotations('rec_b')
    self.assertEqual(
        rec_b_ann,
        [
            hoplite_interface.Annotation(
                id=-1,
                recording_id=-1,
                label='species_a',
                offsets=[1.0, 2.0],
                label_type=hoplite_interface.LabelType.NEGATIVE,
                provenance='agile_metadata',
            )
        ],
    )
    rec_c_ann = md.get_recording_annotations('rec_c')
    self.assertEqual(rec_c_ann, [])

    with self.subTest('no annotations'):
      os.remove(ann_path)
      md = metadata.AgileMetadata.from_directory(self.temp_dir)
      self.assertEqual([], md.get_recording_annotations('rec_a'))


if __name__ == '__main__':
  absltest.main()
