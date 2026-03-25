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

from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import multi_db_impl

from absl.testing import absltest
from absl.testing import parameterized


class MultiDBImplTest(parameterized.TestCase):

  def test_constructor_empty_wrapper(self):
    with self.assertRaisesRegex(
        ValueError, "MultiDBWrapper contains no databases"
    ):
      multi_db_impl.MultiDBWrapper([])

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

    wrapper = multi_db_impl.MultiDBWrapper([db0, db1])

    self.assertEqual(wrapper._split_id(2), (0, 1))
    self.assertEqual(wrapper._split_id(4), (0, 2))
    self.assertEqual(wrapper._split_id(5), (1, 2))
    self.assertEqual(wrapper._split_id(7), (1, 3))
    self.assertEqual(wrapper._split_id(14), (0, 7))
    self.assertEqual(wrapper._split_id(16), (0, 8))

    with self.assertRaises(KeyError):
      wrapper._split_id(0)  # db_index is 0, internal_id is 0 (invalid)
    with self.assertRaises(KeyError):
      wrapper._split_id(22)  # db_index is 0, internal_id is 11 (invalid)
    with self.assertRaises(KeyError):
      wrapper._split_id(1)  # db_index is 1, internal_id is 0 (invalid)
    with self.assertRaises(KeyError):
      wrapper._split_id(13)  # db_index is 1, internal_id is 6 (invalid)

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

    wrapper = multi_db_impl.MultiDBWrapper([db0, db1])

    self.assertEqual(wrapper._split_id(2), (0, 1))
    self.assertEqual(wrapper._split_id(4), (0, 2))
    self.assertEqual(wrapper._split_id(5), (1, 2))
    self.assertEqual(wrapper._split_id(7), (1, 3))
    self.assertEqual(wrapper._split_id(14), (0, 7))
    self.assertEqual(wrapper._split_id(16), (0, 8))

    with self.assertRaises(KeyError):
      wrapper._split_id(6)  # db_index is 0, internal_id is 3 (removed)
    with self.assertRaises(KeyError):
      wrapper._split_id(12)  # db_index is 0, internal_id is 6 (removed)
    with self.assertRaises(KeyError):
      wrapper._split_id(3)  # db_index is 1, internal_id is 1 (removed)
    with self.assertRaises(KeyError):
      wrapper._split_id(11)  # db_index is 1, internal_id is 5 (removed)

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

    wrapper = multi_db_impl.MultiDBWrapper([db0, db1])

    self.assertEqual(wrapper._join_id(0, 1), 2)
    self.assertEqual(wrapper._join_id(0, 2), 4)
    self.assertEqual(wrapper._join_id(1, 2), 5)
    self.assertEqual(wrapper._join_id(1, 3), 7)
    self.assertEqual(wrapper._join_id(0, 7), 14)
    self.assertEqual(wrapper._join_id(0, 8), 16)

    with self.assertRaises(KeyError):
      wrapper._join_id(0, 0)  # db_index is 0, internal_id is 0 (invalid)
    with self.assertRaises(KeyError):
      wrapper._join_id(0, 11)  # db_index is 0, internal_id is 11 (invalid)
    with self.assertRaises(KeyError):
      wrapper._join_id(1, 0)  # db_index is 1, internal_id is 0 (invalid)
    with self.assertRaises(KeyError):
      wrapper._join_id(1, 6)  # db_index is 1, internal_id is 6 (invalid)

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

    wrapper = multi_db_impl.MultiDBWrapper([db0, db1])

    self.assertEqual(wrapper._join_id(0, 1), 2)
    self.assertEqual(wrapper._join_id(0, 2), 4)
    self.assertEqual(wrapper._join_id(1, 2), 5)
    self.assertEqual(wrapper._join_id(1, 3), 7)
    self.assertEqual(wrapper._join_id(0, 7), 14)
    self.assertEqual(wrapper._join_id(0, 8), 16)

    with self.assertRaises(KeyError):
      wrapper._join_id(0, 3)  # db_index is 0, internal_id is 3 (removed)
    with self.assertRaises(KeyError):
      wrapper._join_id(0, 6)  # db_index is 0, internal_id is 6 (removed)
    with self.assertRaises(KeyError):
      wrapper._join_id(1, 1)  # db_index is 1, internal_id is 1 (removed)
    with self.assertRaises(KeyError):
      wrapper._join_id(1, 5)  # db_index is 1, internal_id is 5 (removed)


if __name__ == "__main__":
  absltest.main()
