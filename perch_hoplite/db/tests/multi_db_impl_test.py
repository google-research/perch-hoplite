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

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import multi_db_impl

from absl.testing import absltest
from absl.testing import parameterized


class MultiDBImplTest(parameterized.TestCase):

  def test_constructor_empty_wrapper(self):
    with self.assertRaisesRegex(
        ValueError, "MultiDBWrapper contains no databases"
    ):
      multi_db_impl.MultiDBWrapper({})

  def test_split_id_contiguous_indices(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    dep_id0 = db0.insert_deployment(name="dep0", project="proj")
    rec_id0 = db0.insert_recording(filename="rec0", deployment_id=dep_id0)
    # Add 10 windows to db0: internal IDs 1-10
    for i in range(10):
      db0.insert_window(recording_id=rec_id0, offsets=[float(i)])

    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    dep_id1 = db1.insert_deployment(name="dep1", project="proj")
    rec_id1 = db1.insert_recording(filename="rec1", deployment_id=dep_id1)
    # Add 5 windows to db1: internal IDs 1-5
    for i in range(5):
      db1.insert_window(recording_id=rec_id1, offsets=[float(i)])

    wrapper = multi_db_impl.MultiDBWrapper({"db0": db0, "db1": db1})

    self.assertEqual(wrapper._split_id(2), (0, 1))
    self.assertEqual(wrapper._split_id(4), (0, 2))
    self.assertEqual(wrapper._split_id(5), (1, 2))
    self.assertEqual(wrapper._split_id(7), (1, 3))
    self.assertEqual(wrapper._split_id(14), (0, 7))
    self.assertEqual(wrapper._split_id(16), (0, 8))

  def test_split_id_noncontiguous_indices(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    dep_id0 = db0.insert_deployment(name="dep0", project="proj")
    rec_id0 = db0.insert_recording(filename="rec0", deployment_id=dep_id0)
    # Add 10 windows to db0: internal IDs 1-10
    window_ids_0 = []
    for i in range(10):
      window_ids_0.append(
          db0.insert_window(recording_id=rec_id0, offsets=[float(i)])
      )
    # Remove 2 windows from db0: internal IDs 3 and 6
    db0.remove_window(window_ids_0[2])
    db0.remove_window(window_ids_0[5])

    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    dep_id1 = db1.insert_deployment(name="dep1", project="proj")
    rec_id1 = db1.insert_recording(filename="rec1", deployment_id=dep_id1)
    # Add 5 windows to db1: internal IDs 1-5
    window_ids_1 = []
    for i in range(5):
      window_ids_1.append(
          db1.insert_window(recording_id=rec_id1, offsets=[float(i)])
      )
    # Remove 2 windows from db1: internal IDs 1 and 5
    db1.remove_window(window_ids_1[0])
    db1.remove_window(window_ids_1[4])

    wrapper = multi_db_impl.MultiDBWrapper({"db0": db0, "db1": db1})

    self.assertEqual(wrapper._split_id(2), (0, 1))
    self.assertEqual(wrapper._split_id(4), (0, 2))
    self.assertEqual(wrapper._split_id(5), (1, 2))
    self.assertEqual(wrapper._split_id(7), (1, 3))
    self.assertEqual(wrapper._split_id(14), (0, 7))
    self.assertEqual(wrapper._split_id(16), (0, 8))

  def test_join_id_contiguous_indices(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    dep_id0 = db0.insert_deployment(name="dep0", project="proj")
    rec_id0 = db0.insert_recording(filename="rec0", deployment_id=dep_id0)
    # Add 10 windows to db0: internal IDs 1-10
    for i in range(10):
      db0.insert_window(recording_id=rec_id0, offsets=[float(i)])

    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    dep_id1 = db1.insert_deployment(name="dep1", project="proj")
    rec_id1 = db1.insert_recording(filename="rec1", deployment_id=dep_id1)
    # Add 5 windows to db1: internal IDs 1-5
    for i in range(5):
      db1.insert_window(recording_id=rec_id1, offsets=[float(i)])

    wrapper = multi_db_impl.MultiDBWrapper({"db0": db0, "db1": db1})

    self.assertEqual(wrapper._join_id(0, 1), 2)
    self.assertEqual(wrapper._join_id(0, 2), 4)
    self.assertEqual(wrapper._join_id(1, 2), 5)
    self.assertEqual(wrapper._join_id(1, 3), 7)
    self.assertEqual(wrapper._join_id(0, 7), 14)
    self.assertEqual(wrapper._join_id(0, 8), 16)

  def test_join_id_noncontiguous_indices(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    dep_id0 = db0.insert_deployment(name="dep0", project="proj")
    rec_id0 = db0.insert_recording(filename="rec0", deployment_id=dep_id0)
    window_ids_0 = []
    # Add 10 windows to db0: internal IDs 1-10
    for i in range(10):
      window_ids_0.append(
          db0.insert_window(recording_id=rec_id0, offsets=[float(i)])
      )
    # Remove 2 windows from db0: internal IDs 3 and 6
    db0.remove_window(window_ids_0[2])
    db0.remove_window(window_ids_0[5])

    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    dep_id1 = db1.insert_deployment(name="dep1", project="proj")
    rec_id1 = db1.insert_recording(filename="rec1", deployment_id=dep_id1)
    window_ids_1 = []
    # Add 5 windows to db1: internal IDs 1-5
    for i in range(5):
      window_ids_1.append(
          db1.insert_window(recording_id=rec_id1, offsets=[float(i)])
      )
    # Remove 2 windows from db1: internal IDs 1 and 5
    db1.remove_window(window_ids_1[0])
    db1.remove_window(window_ids_1[4])

    wrapper = multi_db_impl.MultiDBWrapper({"db0": db0, "db1": db1})

    self.assertEqual(wrapper._join_id(0, 1), 2)
    self.assertEqual(wrapper._join_id(0, 2), 4)
    self.assertEqual(wrapper._join_id(1, 2), 5)
    self.assertEqual(wrapper._join_id(1, 3), 7)
    self.assertEqual(wrapper._join_id(0, 7), 14)
    self.assertEqual(wrapper._join_id(0, 8), 16)

  def test_create(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    wrapper = multi_db_impl.MultiDBWrapper.create(dbs={"db0": db0, "db1": db1})
    self.assertIsInstance(wrapper, multi_db_impl.MultiDBWrapper)
    self.assertLen(wrapper._dbs, 2)

  def test_extra_table_columns(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    wrapper = multi_db_impl.MultiDBWrapper.create(dbs={"db0": db0, "db1": db1})

    # Initially empty
    self.assertEqual(wrapper.get_extra_table_columns()["deployments"], {})

    # Add to wrapper (propagates to both)
    wrapper.add_extra_table_column("deployments", "extra0", str)
    self.assertEqual(
        wrapper.get_extra_table_columns()["deployments"], {"extra0": str}
    )
    self.assertEqual(
        db0.get_extra_table_columns()["deployments"], {"extra0": str}
    )
    self.assertEqual(
        db1.get_extra_table_columns()["deployments"], {"extra0": str}
    )

    # Intersection logic: add to only one DB
    db0.add_extra_table_column("deployments", "extra1", int)
    self.assertEqual(
        wrapper.get_extra_table_columns()["deployments"], {"extra0": str}
    )

    # Add to the other DB
    db1.add_extra_table_column("deployments", "extra1", int)
    self.assertEqual(
        wrapper.get_extra_table_columns()["deployments"],
        {"extra0": str, "extra1": int},
    )

    # Mismatched types should not be in intersection
    db0.add_extra_table_column("deployments", "extra2", float)
    db1.add_extra_table_column("deployments", "extra2", int)
    self.assertEqual(
        wrapper.get_extra_table_columns()["deployments"],
        {"extra0": str, "extra1": int},
    )

  def test_metadata(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    wrapper = multi_db_impl.MultiDBWrapper.create(dbs={"db0": db0, "db1": db1})

    meta0 = config_dict.ConfigDict({"a": 1})
    meta1 = config_dict.ConfigDict({"b": 2})

    wrapper.insert_metadata("db0/k0", meta0)
    wrapper.insert_metadata("db1/k1", meta1)

    self.assertEqual(wrapper.get_metadata("db0/k0"), meta0)
    self.assertEqual(wrapper.get_metadata("db1/k1"), meta1)

    # Test get_metadata(None)
    all_meta = wrapper.get_metadata(None)
    self.assertEqual(all_meta["db0/k0"], meta0)
    self.assertEqual(all_meta["db1/k1"], meta1)

    # Test individual removal
    wrapper.remove_metadata("db0/k0")
    with self.assertRaises(KeyError):
      wrapper.get_metadata("db0/k0")
    self.assertEqual(wrapper.get_metadata("db1/k1"), meta1)

    # Test removal of all
    wrapper.insert_metadata("db0/k0", meta0)
    wrapper.remove_metadata(None)
    self.assertEmpty(wrapper.get_metadata(None))

  def test_metadata_invalid_keys(self):
    wrapper = multi_db_impl.MultiDBWrapper.create(
        dbs={"db0": in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)}
    )

    with self.assertRaisesRegex(ValueError, "format 'db_name/internal_key'"):
      wrapper.insert_metadata("no_slash", config_dict.ConfigDict())

    with self.assertRaisesRegex(ValueError, "Database db1 not found"):
      wrapper.insert_metadata("db1/key", config_dict.ConfigDict())


class MultiDBGetAllTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Create two in-memory databases
    self.db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=2)
    self.db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=2)
    self.dbs = {"db0": self.db0, "db1": self.db1}
    self.multi_db = multi_db_impl.MultiDBWrapper(self.dbs)

    # Populate DB0
    self.d0 = self.db0.insert_deployment(name="D0", project="P0")
    self.r0 = self.db0.insert_recording(filename="R0", deployment_id=self.d0)
    self.w0 = self.db0.insert_window(
        recording_id=self.r0, offsets=[0.0, 1.0], embedding=np.array([1.0, 0.0])
    )
    self.a0 = self.db0.insert_annotation(
        recording_id=self.r0,
        offsets=[0.0, 1.0],
        label="L0",
        label_type=datatypes.LabelType.POSITIVE,
        provenance="prov0",
    )

    # Populate DB1
    self.d1 = self.db1.insert_deployment(name="D1", project="P1")
    self.r1 = self.db1.insert_recording(filename="R1", deployment_id=self.d1)
    self.w1 = self.db1.insert_window(
        recording_id=self.r1, offsets=[2.0, 3.0], embedding=np.array([0.0, 1.0])
    )
    self.a1 = self.db1.insert_annotation(
        recording_id=self.r1,
        offsets=[2.0, 3.0],
        label="L1",
        label_type=datatypes.LabelType.NEGATIVE,
        provenance="prov1",
    )

  def test_get_all_deployments(self):
    # All deployments
    deployments = self.multi_db.get_all_deployments()
    self.assertLen(deployments, 2)
    # Since d0 is 1 and d1 is 1, the external ID for d0 is 1*2 + 0 = 2, and the
    # external ID for d1 is 1*2 + 1 = 3
    ids = {deployment.id for deployment in deployments}
    self.assertIn(2, ids)
    self.assertIn(3, ids)

    # Expected to filter for d1
    name_filter = config_dict.ConfigDict({"eq": {"name": "D1"}})
    deployments = self.multi_db.get_all_deployments(filter=name_filter)
    self.assertLen(deployments, 1)
    self.assertEqual(deployments[0].name, "D1")
    self.assertEqual(deployments[0].id, 3)

    # Expected to filter for d0
    id_filter = config_dict.ConfigDict({"eq": {"id": 2}})
    deployments = self.multi_db.get_all_deployments(filter=id_filter)
    self.assertLen(deployments, 1)
    self.assertEqual(deployments[0].name, "D0")

  def test_get_all_recordings(self):
    # All recordings
    recordings = self.multi_db.get_all_recordings()
    self.assertLen(recordings, 2)

    # Expected to filter for r1
    deployment_filter = config_dict.ConfigDict({"eq": {"deployment_id": 3}})
    recordings = self.multi_db.get_all_recordings(filter=deployment_filter)
    self.assertLen(recordings, 1)
    self.assertEqual(recordings[0].filename, "R1")

  def test_get_all_windows(self):
    # All windows
    windows = self.multi_db.get_all_windows()
    self.assertLen(windows, 2)

    # Expected to filter for w0
    recording_filter = config_dict.ConfigDict({"eq": {"recording_id": 2}})
    windows = self.multi_db.get_all_windows(filter=recording_filter)
    self.assertLen(windows, 1)
    self.assertEqual(windows[0].offsets, [0.0, 1.0])

    # Expected to filter for w1
    deployment_filter = config_dict.ConfigDict({"eq": {"name": "D1"}})
    windows = self.multi_db.get_all_windows(
        deployments_filter=deployment_filter
    )
    self.assertLen(windows, 1)
    self.assertEqual(windows[0].offsets, [2.0, 3.0])

  def test_match_window_ids(self):
    # All windows
    ids = self.multi_db.match_window_ids()
    self.assertLen(ids, 2)

    # Expected to filter for w1
    window_filter = config_dict.ConfigDict({"eq": {"id": 3}})
    ids = self.multi_db.match_window_ids(windows_filter=window_filter)
    self.assertLen(ids, 1)
    self.assertEqual(ids[0], 3)

    # Expected to filter for w0
    deployment_filter = config_dict.ConfigDict({"eq": {"project": "P0"}})
    ids = self.multi_db.match_window_ids(deployments_filter=deployment_filter)
    self.assertLen(ids, 1)
    self.assertEqual(ids[0], 2)

  def test_get_all_annotations(self):
    annotations = self.multi_db.get_all_annotations()
    self.assertLen(annotations, 2)

    # Expected to filter for w1
    id_filter = config_dict.ConfigDict({"eq": {"id": 3}})
    annotations = self.multi_db.get_all_annotations(filter=id_filter)
    self.assertLen(annotations, 1)
    self.assertEqual(annotations[0].label, "L1")


if __name__ == "__main__":
  absltest.main()
