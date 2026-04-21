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

  def test_insert_deployment_invalid_project(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    db0.insert_deployment(name="d0", project="p0")
    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    db1.insert_deployment(name="d1", project="p1")

    wrapper = multi_db_impl.MultiDBWrapper.create(dbs={"db0": db0, "db1": db1})

    # Project in zero DBs
    with self.assertRaisesRegex(NotImplementedError, "must exist in exactly 1"):
      wrapper.insert_deployment(name="d2", project="p2")

    # Project in multiple DBs
    db1.insert_deployment(name="d0_dup", project="p0")
    with self.assertRaisesRegex(NotImplementedError, "must exist in exactly 1"):
      wrapper.insert_deployment(name="d0_new", project="p0")


class MultiDBIdTranslationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    self.db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    self.wrapper = multi_db_impl.MultiDBWrapper(
        {"db0": self.db0, "db1": self.db1}
    )

    self.internal_dep0_id = self.db0.insert_deployment(
        name="dep0", project="p0"
    )  # 1
    self.external_dep0_id = self.wrapper._join_id(0, self.internal_dep0_id)  # 2
    self.internal_dep1_id = self.db1.insert_deployment(
        name="dep1", project="p1"
    )  # 1
    self.external_dep1_id = self.wrapper._join_id(1, self.internal_dep1_id)  # 3

    self.internal_rec0_id = self.db0.insert_recording(
        filename="rec0", deployment_id=self.internal_dep0_id
    )  # 1
    self.external_rec0_id = self.wrapper._join_id(0, self.internal_rec0_id)  # 2
    self.internal_rec1_id = self.db1.insert_recording(
        filename="rec1", deployment_id=self.internal_dep1_id
    )  # 1
    self.external_rec1_id = self.wrapper._join_id(1, self.internal_rec1_id)  # 3

    self.internal_win0_id = self.db0.insert_window(
        recording_id=self.internal_rec0_id, offsets=[0.0, 1.0]
    )  # 1
    self.external_win0_id = self.wrapper._join_id(0, self.internal_win0_id)  # 2
    self.internal_win1_id = self.db1.insert_window(
        recording_id=self.internal_rec1_id, offsets=[2.0, 3.0]
    )  # 1
    self.external_win1_id = self.wrapper._join_id(1, self.internal_win1_id)  # 3

    self.internal_ann0_id = self.db0.insert_annotation(
        recording_id=self.internal_rec0_id,
        offsets=[0.0, 1.0],
        label="L",
        label_type=datatypes.LabelType.POSITIVE,
        provenance="P",
    )  # 1
    self.external_ann0_id = self.wrapper._join_id(0, self.internal_ann0_id)  # 2
    self.internal_ann1_id = self.db1.insert_annotation(
        recording_id=self.internal_rec1_id,
        offsets=[2.0, 3.0],
        label="M",
        label_type=datatypes.LabelType.NEGATIVE,
        provenance="Q",
    )  # 1
    self.external_ann1_id = self.wrapper._join_id(1, self.internal_ann1_id)  # 3

    self.internal_ann2_id = self.db1.insert_annotation(
        recording_id=self.internal_rec1_id,
        offsets=[2.5, 3.5],
        label="N",
        label_type=datatypes.LabelType.POSITIVE,
        provenance="P",
    )  # 2
    self.external_ann2_id = self.wrapper._join_id(1, self.internal_ann2_id)  # 5

  def test_get_deployment_translation(self):
    dep0 = self.wrapper.get_deployment(self.external_dep0_id)
    self.assertEqual(dep0.id, self.external_dep0_id)
    dep1 = self.wrapper.get_deployment(self.external_dep1_id)
    self.assertEqual(dep1.id, self.external_dep1_id)

  def test_get_recording_translation(self):
    rec0 = self.wrapper.get_recording(self.external_rec0_id)
    self.assertEqual(rec0.id, self.external_rec0_id)
    self.assertEqual(rec0.deployment_id, self.external_dep0_id)
    rec1 = self.wrapper.get_recording(self.external_rec1_id)
    self.assertEqual(rec1.id, self.external_rec1_id)
    self.assertEqual(rec1.deployment_id, self.external_dep1_id)

  def test_get_window_translation(self):
    win0 = self.wrapper.get_window(self.external_win0_id)
    self.assertEqual(win0.id, self.external_win0_id)
    self.assertEqual(win0.recording_id, self.external_rec0_id)
    win1 = self.wrapper.get_window(self.external_win1_id)
    self.assertEqual(win1.id, self.external_win1_id)
    self.assertEqual(win1.recording_id, self.external_rec1_id)

  def test_get_annotation_translation(self):
    ann0 = self.wrapper.get_annotation(self.external_ann0_id)
    self.assertEqual(ann0.id, self.external_ann0_id)
    self.assertEqual(ann0.recording_id, self.external_rec0_id)
    ann1 = self.wrapper.get_annotation(self.external_ann1_id)
    self.assertEqual(ann1.id, self.external_ann1_id)
    self.assertEqual(ann1.recording_id, self.external_rec1_id)

  def test_get_window_annotations_translation(self):
    anns0 = self.wrapper.get_window_annotations(self.external_win0_id)
    self.assertLen(anns0, 1)
    self.assertEqual(anns0[0].id, self.external_ann0_id)
    self.assertEqual(anns0[0].recording_id, self.external_rec0_id)
    anns1 = self.wrapper.get_window_annotations(self.external_win1_id)
    self.assertLen(anns1, 2)
    ids1 = {ann.id for ann in anns1}
    self.assertIn(self.external_ann1_id, ids1)
    self.assertIn(self.external_ann2_id, ids1)
    for ann in anns1:
      self.assertEqual(ann.recording_id, self.external_rec1_id)


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

    # Explicit window filter (matching on offsets)
    win_filter = config_dict.ConfigDict({"approx": {"offsets": [0.0, 1.0]}})
    windows = self.multi_db.get_all_windows(filter=win_filter)
    self.assertLen(windows, 1)
    self.assertEqual(windows[0].id, 2)

  def test_get_all_windows_complex_filters(self):
    # Filter by deployment project and recording ID
    # w0: D0 (project is P0), R0 (external ID is 2)
    # w1: D1 (project is P1), R1 (external ID is 3)
    dep_filter = config_dict.ConfigDict({"eq": {"project": "P0"}})
    rec_filter = config_dict.ConfigDict({"eq": {"id": 2}})
    windows = self.multi_db.get_all_windows(
        deployments_filter=dep_filter, recordings_filter=rec_filter
    )
    self.assertLen(windows, 1)
    self.assertEqual(windows[0].id, 2)

    # Contradictory filters
    dep_filter = config_dict.ConfigDict({"eq": {"project": "P1"}})
    rec_filter = config_dict.ConfigDict({"eq": {"id": 2}})
    windows = self.multi_db.get_all_windows(
        deployments_filter=dep_filter, recordings_filter=rec_filter
    )
    self.assertEmpty(windows)

    # Two components match everything but one matches nothing
    dep_filter = config_dict.ConfigDict({"isin": {"project": ["P0", "P1"]}})
    rec_filter = config_dict.ConfigDict({"isin": {"filename": ["R0", "R1"]}})
    ann_filter = config_dict.ConfigDict({"eq": {"label": "NON_EXISTENT"}})
    windows = self.multi_db.get_all_windows(
        deployments_filter=dep_filter,
        recordings_filter=rec_filter,
        annotations_filter=ann_filter,
    )
    self.assertEmpty(windows)

  def test_get_all_windows_annotations_filter(self):
    # Filter windows by annotation label
    # a0 (label L0) is on r0
    # a1 (label L1) is on r1
    ann_filter = config_dict.ConfigDict({"eq": {"label": "L0"}})
    windows = self.multi_db.get_all_windows(annotations_filter=ann_filter)
    self.assertLen(windows, 1)
    self.assertEqual(windows[0].id, 2)

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

  def test_match_window_ids_limit(self):
    ids = self.multi_db.match_window_ids(limit=1)
    self.assertLen(ids, 1)


class MultiDBAggregationTest(parameterized.TestCase):

  def test_aggregations(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=2)
    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=2)
    wrapper = multi_db_impl.MultiDBWrapper({"db0": db0, "db1": db1})

    # Projects
    db0.insert_deployment(name="d0", project="p0")
    db1.insert_deployment(name="d1", project="p1")
    self.assertEqual(wrapper.get_all_projects(), ["p0", "p1"])

    # Labels and Counts
    r0 = db0.insert_recording(filename="r0", deployment_id=1)
    db0.insert_annotation(
        recording_id=r0,
        offsets=[0, 1],
        label="L",
        label_type=datatypes.LabelType.POSITIVE,
        provenance="P",
    )
    r1 = db1.insert_recording(filename="r1", deployment_id=1)
    db1.insert_annotation(
        recording_id=r1,
        offsets=[0, 1],
        label="L",
        label_type=datatypes.LabelType.POSITIVE,
        provenance="P",
    )
    db1.insert_annotation(
        recording_id=r1,
        offsets=[2, 3],
        label="M",
        label_type=datatypes.LabelType.NEGATIVE,
        provenance="P",
    )
    self.assertEqual(wrapper.get_all_labels(), ["L", "M"])
    self.assertEqual(
        wrapper.get_all_labels(label_type=datatypes.LabelType.POSITIVE), ["L"]
    )
    self.assertEqual(
        wrapper.get_all_labels(label_type=datatypes.LabelType.NEGATIVE), ["M"]
    )
    self.assertEqual(wrapper.count_each_label(), {"L": 2, "M": 1})
    self.assertEqual(
        wrapper.count_each_label(label_type=datatypes.LabelType.POSITIVE),
        {"L": 2},
    )
    self.assertEqual(
        wrapper.count_each_label(label_type=datatypes.LabelType.NEGATIVE),
        {"M": 1},
    )

    # Embeddings
    db0.insert_window(recording_id=r0, offsets=[0, 1], embedding=np.zeros(2))
    db1.insert_window(recording_id=r1, offsets=[0, 1], embedding=np.zeros(2))
    self.assertEqual(wrapper.count_embeddings(), 2)
    self.assertEqual(wrapper.get_embedding_dim(), 2)
    self.assertEqual(wrapper.get_embedding_dtype(), np.float16)


class MultiDBLifecycleTest(parameterized.TestCase):

  def test_lifecycle_methods(self):
    # This tests that the functions don't crash
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    wrapper = multi_db_impl.MultiDBWrapper({"db0": db0})
    wrapper.commit()
    wrapper.rollback()

    split_wrapper = wrapper.thread_split()
    self.assertIsInstance(split_wrapper, multi_db_impl.MultiDBWrapper)
    self.assertLen(split_wrapper.get_dbs(), 1)

  def test_removal_methods(self):
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=1)
    wrapper = multi_db_impl.MultiDBWrapper({"db0": db0, "db1": db1})
    db0_dep_id = db0.insert_deployment(name="initial", project="p")  # 1

    dep_id = wrapper.insert_deployment(name="d", project="p")  # 4
    rec_id = wrapper.insert_recording(filename="r", deployment_id=dep_id)  # 2
    win_id = wrapper.insert_window(recording_id=rec_id, offsets=[0, 1])  # 2
    ann_id = wrapper.insert_annotation(
        recording_id=rec_id,
        offsets=[0, 1],
        label="L",
        label_type=datatypes.LabelType.POSITIVE,
        provenance="P",
    )  # 2

    self.assertLen(db0.get_all_annotations(), 1)
    self.assertLen(wrapper.get_all_annotations(), 1)
    wrapper.remove_annotation(ann_id)
    self.assertEmpty(db0.get_all_annotations())
    self.assertEmpty(wrapper.get_all_annotations())

    self.assertLen(db0.get_all_windows(), 1)
    self.assertLen(wrapper.get_all_windows(), 1)
    wrapper.remove_window(win_id)
    self.assertEmpty(db0.get_all_windows())
    self.assertEmpty(wrapper.get_all_windows())

    self.assertLen(db0.get_all_recordings(), 1)
    self.assertLen(wrapper.get_all_recordings(), 1)
    wrapper.remove_recording(rec_id)
    self.assertEmpty(db0.get_all_recordings())
    self.assertEmpty(wrapper.get_all_recordings())

    self.assertLen(db0.get_all_deployments(), 2)
    self.assertLen(wrapper.get_all_deployments(), 2)
    wrapper.remove_deployment(dep_id)
    self.assertLen(db0.get_all_deployments(), 1)
    self.assertLen(wrapper.get_all_deployments(), 1)

    # Remove the initial deployment correctly by joining its internal ID
    wrapper.remove_deployment(wrapper._join_id(0, db0_dep_id))
    self.assertEmpty(db0.get_all_deployments())
    self.assertEmpty(wrapper.get_all_deployments())


if __name__ == "__main__":
  absltest.main()
