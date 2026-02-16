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

"""Tests for the embeddings databases interface."""

from perch_hoplite.db import interface

from absl.testing import absltest


class DynamicInfoTest(absltest.TestCase):

  def test_dynamic_attributes(self):
    info = interface.DynamicInfo(
        foo="bar",
        bar="baz",
        value=100,
    )
    self.assertEqual(info.foo, "bar")
    self.assertEqual(info.bar, "baz")
    self.assertEqual(info.value, 100)
    self.assertRaises(AttributeError, lambda: info.status)
    info.status = "active"
    self.assertEqual(info.status, "active")

  def test_repr(self):
    info = interface.DynamicInfo(
        foo="bar",
        bar="baz",
        value=100,
    )
    self.assertEqual(
        repr(info),
        "DynamicInfo(foo='bar', bar='baz', value=100)",
    )
    info.status = "active"
    self.assertEqual(
        repr(info),
        "DynamicInfo(foo='bar', bar='baz', value=100, status='active')",
    )

  def test_eq(self):
    info1 = interface.DynamicInfo(
        foo="bar",
        bar="baz",
        value=100,
    )

    info2 = interface.DynamicInfo(
        foo="bar",
        bar="baz",
        value=100,
    )
    self.assertEqual(info1, info2)

    info3 = interface.DynamicInfo(
        foo="bar",
        bar="baz",
        value=101,
    )
    self.assertNotEqual(info1, info3)

    info4 = interface.DynamicInfo(
        foo="bar",
        bar="baz",
        value=100,
        status="active",
    )
    self.assertNotEqual(info1, info4)

    self.assertNotEqual(info1, "some string")


class DeploymentTest(absltest.TestCase):

  def test_required_attributes(self):
    dep = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
    )
    self.assertEqual(dep.id, 101)
    self.assertEqual(dep.name, "Sensor-A")
    self.assertEqual(dep.project, "River Monitoring")
    self.assertIsNone(dep.latitude)
    self.assertIsNone(dep.longitude)

  def test_optional_attributes(self):
    dep = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
        latitude=51.51,
        longitude=0.0,
    )
    self.assertAlmostEqual(dep.latitude, 51.51)
    self.assertAlmostEqual(dep.longitude, 0.0)

  def test_dynamic_attributes(self):
    dep = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
        latitude=51.51,
        status="active",
    )
    self.assertEqual(dep.status, "active")
    self.assertRaises(AttributeError, lambda: dep.country)
    dep.country = "United Kingdom"
    self.assertEqual(dep.country, "United Kingdom")

  def test_missing_required_argument(self):
    self.assertRaises(TypeError, lambda: interface.Deployment(name="Sensor-A"))
    self.assertRaises(
        TypeError, lambda: interface.Deployment(id=101, name="Sensor-A")
    )

  def test_repr(self):
    dep = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
    )
    self.assertEqual(
        repr(dep),
        "Deployment(id=101, name='Sensor-A', project='River Monitoring',"
        " latitude=None, longitude=None)",
    )
    dep.latitude = 51.51
    dep.country = "United Kingdom"
    self.assertEqual(
        repr(dep),
        "Deployment(id=101, name='Sensor-A', project='River Monitoring',"
        " latitude=51.51, longitude=None, country='United Kingdom')",
    )

  def test_eq(self):
    dep1 = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
    )

    dep2 = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
    )
    self.assertEqual(dep1, dep2)

    dep3 = interface.Deployment(
        id=102,
        name="Sensor-B",
        project="River Monitoring",
    )
    self.assertNotEqual(dep1, dep3)

    dep4 = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
        latitude=51.51,
    )
    self.assertNotEqual(dep1, dep4)

    dep5 = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
        status="active",
    )
    self.assertNotEqual(dep1, dep5)

    dep6 = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
        status="active",
    )
    self.assertEqual(dep5, dep6)

    dep7 = interface.Deployment(
        id=101,
        name="Sensor-A",
        project="River Monitoring",
        status="inactive",
    )
    self.assertNotEqual(dep6, dep7)

    self.assertNotEqual(dep1, "some string")


if __name__ == "__main__":
  absltest.main()
