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

import dataclasses
from perch_hoplite.db import datatypes
from absl.testing import absltest


class DatatypesTest(absltest.TestCase):

  def test_dynamic_info_with_dataclasses_replace(self):
    deployment = datatypes.Deployment(
        id=1, name='test', project='p', extra_field='extra'
    )
    self.assertEqual(deployment.extra_field, 'extra')
    # Using the dataclasses.replace method
    # This is expected to lose extra_field because it's not a formal field and
    # dataclasses.replace is used
    new_deployment = dataclasses.replace(deployment, id=2)
    self.assertEqual(new_deployment.id, 2)
    self.assertEqual(new_deployment.name, 'test')
    self.assertEqual(new_deployment.project, 'p')
    with self.assertRaises(AttributeError):
      _ = new_deployment.extra_field

  def test_dynamic_info_with_dynamic_info_replace(self):
    deployment = datatypes.Deployment(
        id=1, name='test', project='p', extra_field='extra'
    )
    # Using the DynamicInfo.replace method
    # This is expected to keep extra_field because DynamicInfo.replace is used
    new_deployment = deployment.replace(id=2)
    self.assertEqual(new_deployment.id, 2)
    self.assertEqual(new_deployment.name, 'test')
    self.assertEqual(new_deployment.project, 'p')
    self.assertEqual(new_deployment.extra_field, 'extra')


if __name__ == '__main__':
  absltest.main()
