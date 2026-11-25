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

"""Tests for sql_utils and optional/lazy USearch behavior."""

import datetime as dt

from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import sql_utils
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.db import usearch_store

from absl.testing import absltest
from absl.testing import parameterized


class SqlUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple', 'col_name', True),
      ('leading_underscore', '_col123', True),
      ('empty', '', False),
      ('starts_with_digit', '1col', False),
      ('contains_space', 'col name', False),
      ('sql_injection', 'col; DROP TABLE windows', False),
  )
  def test_is_valid_sql_identifier(self, name: str, expected: bool):
    self.assertEqual(sql_utils.is_valid_sql_identifier(name), expected)

  @parameterized.named_parameters(
      ('simple', 'crow', '"crow"'),
      (
          'spaces_and_hyphen',
          'American Redstart - song',
          '"American Redstart - song"',
      ),
      ('embedded_quote', 'a"b', '"a""b"'),
  )
  def test_quote_sql_identifier(self, raw: str, expected: str):
    self.assertEqual(sql_utils.quote_sql_identifier(raw), expected)

  def test_empty_isin_and_notin(self):
    cond_in, vals_in = sql_utils.format_sql_where_conditions(
        config_dict.create(isin=dict(project=[]))
    )
    self.assertEqual(cond_in, '1 = 0')
    self.assertEmpty(vals_in)

    cond_notin, vals_notin = sql_utils.format_sql_where_conditions(
        config_dict.create(notin=dict(project=[]))
    )
    self.assertEqual(cond_notin, '1 = 1')
    self.assertEmpty(vals_notin)

  def test_custom_offsets_approx_fn(self):
    def custom_approx(col: str, val):
      return f'LEN({col}) = LEN(?)', [val]

    cond, vals = sql_utils.format_sql_where_conditions(
        config_dict.create(approx=dict(offsets=[1.0, 2.0])),
        table_prefix='windows',
        offsets_approx_fn=custom_approx,
    )
    self.assertEqual(cond, 'LEN(windows.offsets) = LEN(?)')
    self.assertEqual(vals, [[1.0, 2.0]])

  def test_normalize_sql_value(self):
    ts = dt.datetime(2025, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
    self.assertEqual(sql_utils.normalize_sql_value(ts), ts.isoformat())
    self.assertEqual(
        sql_utils.normalize_sql_value(datatypes.LabelType.POSITIVE), 1
    )
    self.assertEqual(sql_utils.normalize_sql_value(np.int64(7)), 7)
    self.assertEqual(sql_utils.normalize_sql_value(np.float32(1.5)), 1.5)


class OptionalUSearchSQLiteTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = epath.Path(self.create_tempdir().full_path)

  def test_typo_path_raises_without_creating_directory(self):
    missing_dir = self.tempdir / 'does_not_exist'
    with self.assertRaisesRegex(ValueError, 'No usearch_cfg was found'):
      sqlite_usearch_impl.SQLiteUSearchDB.create(missing_dir.as_posix())
    self.assertFalse(missing_dir.exists())

  def test_lazy_usearch_index_not_created_until_embeddings_written(self):
    db_dir = self.tempdir / 'lazy_db'
    cfg = sqlite_usearch_impl.get_default_usearch_config(embedding_dim=16)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_dir.as_posix(), usearch_cfg=cfg
    )
    dep_id = db.insert_deployment(name='d1', project='p1')
    rec_id = db.insert_recording(filename='a.wav', deployment_id=dep_id)
    db.insert_annotation(
        recording_id=rec_id,
        offsets=[0.0, 5.0],
        label='bird',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='model',
    )
    db.commit()

    self.assertEqual(db.count_embeddings(), 0)
    self.assertFalse(db.usearch_path.exists())
    self.assertEmpty(
        db.search(
            np.zeros(16, dtype=np.float32), search_list_size=5
        ).search_results
    )
    db.close()

    # Read-only reopen succeeds even with no usearch.index on disk yet.
    ro_db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_dir.as_posix(), readonly=True
    )
    self.assertEqual(ro_db.count_embeddings(), 0)
    self.assertEqual(ro_db.get_embedding_dim(), 16)
    self.assertEqual(ro_db.get_embedding_dtype(), np.float16)
    self.assertLen(ro_db.get_all_annotations(), 1)
    ro_db.close()

    # Writing an embedding lazily materializes usearch.index on commit.
    rw_db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_dir.as_posix())
    vec = np.ones(16, dtype=np.float32)
    wid = rw_db.insert_window(
        recording_id=rec_id, offsets=[0.0, 5.0], embedding=vec
    )
    rw_db.commit()
    self.assertTrue(rw_db.usearch_path.exists())
    self.assertEqual(rw_db.count_embeddings(), 1)
    np.testing.assert_allclose(rw_db.get_embedding(wid), vec.astype(np.float16))
    # Backwards-compatible `db.ui` property works directly.
    np.testing.assert_allclose(rw_db.ui.get(wid), vec.astype(np.float16))
    rw_db.close()

  def test_create_without_embeddings_lifecycle(self):
    db_dir = self.tempdir / 'no_emb_db'
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_dir.as_posix(), use_embeddings=False
    )
    self.assertFalse(db.usearch_path.exists())
    self.assertEqual(db.count_embeddings(), 0)

    dep_id = db.insert_deployment(name='site1', project='proj1')
    rec_id = db.insert_recording(filename='f1.wav', deployment_id=dep_id)
    ann_id = db.insert_annotation(
        recording_id=rec_id,
        offsets=[1.0, 2.0],
        label='owl',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='human',
    )
    # Inserting a window without an embedding works.
    wid = db.insert_window(
        recording_id=rec_id, offsets=[0.0, 5.0], embedding=None
    )
    db.commit()

    # Inserting an embedding fails cleanly before mutating SQL.
    with self.assertRaisesRegex(ValueError, 'without an embedding index'):
      db.insert_window(
          recording_id=rec_id,
          offsets=[5.0, 10.0],
          embedding=np.zeros(8, dtype=np.float32),
      )
    self.assertLen(db.get_all_windows(), 1)

    with self.assertRaisesRegex(ValueError, 'Embeddings are not enabled'):
      db.get_embedding_dim()
    with self.assertRaisesRegex(ValueError, 'Embeddings are not enabled'):
      db.search(np.zeros(8, dtype=np.float32), search_list_size=3)
    with self.assertRaisesRegex(KeyError, 'Embedding vector not found'):
      db.get_embedding(wid)

    # thread_split() works cleanly on a no-embeddings DB.
    split_db = db.thread_split()
    self.assertEqual(split_db.count_embeddings(), 0)
    self.assertEqual(split_db.get_annotation(ann_id).label, 'owl')
    split_db.close()

    # Removing window/recording/deployment works without a USearch index.
    db.remove_window(wid)
    db.commit()
    self.assertEmpty(db.get_all_windows())
    db.close()

    # Reopening without repeating `use_embeddings=False` succeeds automatically.
    reopened = sqlite_usearch_impl.SQLiteUSearchDB.create(db_dir.as_posix())
    self.assertEqual(reopened.count_embeddings(), 0)
    self.assertFalse(reopened.usearch_path.exists())
    self.assertEqual(reopened.get_recording(rec_id).filename, 'f1.wav')
    reopened.close()

    # Read-only reopen also succeeds.
    ro_db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_dir.as_posix(), readonly=True
    )
    self.assertEqual(ro_db.count_embeddings(), 0)
    ro_db.close()

  def test_upgrade_no_embeddings_db_by_passing_usearch_cfg(self):
    db_dir = self.tempdir / 'upgrade_db'
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_dir.as_posix(), use_embeddings=False
    )
    dep_id = db.insert_deployment(name='d1', project='p1')
    rec_id = db.insert_recording(filename='r1.wav', deployment_id=dep_id)
    db.commit()
    db.close()

    cfg = usearch_store.get_default_usearch_config(embedding_dim=8)
    with self.assertRaisesRegex(ValueError, 'read-only'):
      sqlite_usearch_impl.SQLiteUSearchDB.create(
          db_dir.as_posix(), usearch_cfg=cfg, readonly=True
      )

    upgraded = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_dir.as_posix(), usearch_cfg=cfg
    )
    vec = np.arange(8, dtype=np.float32)
    wid = upgraded.insert_window(
        recording_id=rec_id, offsets=[0.0, 5.0], embedding=vec
    )
    upgraded.commit()
    self.assertEqual(upgraded.count_embeddings(), 1)
    np.testing.assert_allclose(
        upgraded.get_embedding(wid), vec.astype(np.float16)
    )
    upgraded.close()

  def test_conflicting_use_embeddings_flags_raise(self):
    db_dir = self.tempdir / 'conflict_db'
    cfg = usearch_store.get_default_usearch_config(embedding_dim=8)
    with self.assertRaisesRegex(ValueError, 'Cannot specify both'):
      sqlite_usearch_impl.SQLiteUSearchDB.create(
          db_dir.as_posix(), usearch_cfg=cfg, use_embeddings=False
      )

  def test_open_embeddings_db_with_use_embeddings_false_warns(self):
    db_dir = self.tempdir / 'warn_db'
    cfg = usearch_store.get_default_usearch_config(embedding_dim=8)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_dir.as_posix(), usearch_cfg=cfg
    )
    dep_id = db.insert_deployment(name='d1', project='p1')
    rec_id = db.insert_recording(filename='r1.wav', deployment_id=dep_id)
    vec = np.ones(8, dtype=np.float32)
    wid = db.insert_window(
        recording_id=rec_id, offsets=[0.0, 5.0], embedding=vec
    )
    db.commit()
    db.close()

    with self.assertLogs(level='WARNING') as log_context:
      no_emb_db = sqlite_usearch_impl.SQLiteUSearchDB.create(
          db_dir.as_posix(), use_embeddings=False
      )
    self.assertIn(
        'Opening an embeddings-enabled database with `use_embeddings=False`',
        '\n'.join(log_context.output),
    )
    self.assertEqual(no_emb_db.count_embeddings(), 0)
    self.assertEqual(no_emb_db.get_recording(rec_id).filename, 'r1.wav')
    # Does not overwrite DB metadata with `use_embeddings=False`.
    self.assertNotIn(usearch_store.DB_OPTIONS_KEY, no_emb_db.get_metadata(None))
    with self.assertLogs(level='WARNING'):
      split_db = no_emb_db.thread_split()
    self.assertEqual(split_db.count_embeddings(), 0)
    split_db.close()
    no_emb_db.close()

    # Reopening normally still loads the existing embeddings cleanly.
    reopened = sqlite_usearch_impl.SQLiteUSearchDB.create(db_dir.as_posix())
    self.assertEqual(reopened.count_embeddings(), 1)
    np.testing.assert_allclose(
        reopened.get_embedding(wid), vec.astype(np.float16)
    )
    reopened.close()

  def test_backwards_compatible_reexports(self):
    self.assertIs(
        sqlite_usearch_impl.get_default_usearch_config,
        usearch_store.get_default_usearch_config,
    )
    self.assertIs(
        sqlite_usearch_impl.format_sql_where_conditions,
        sql_utils.format_sql_where_conditions,
    )
    self.assertIs(
        sqlite_usearch_impl.is_valid_sql_identifier,
        sql_utils.is_valid_sql_identifier,
    )


if __name__ == '__main__':
  absltest.main()
