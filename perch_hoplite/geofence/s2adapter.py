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

# pytype: skip-file
"""Adapter to bridge differences between internal S2 and external s2geometry."""

# pylint: disable=protected-access

import s2geometry as _s2


class CallableFloat(float):

  def __call__(self):
    return self


class S1Angle:
  """Adapter for S1Angle."""

  def __init__(self, angle_obj):
    self._angle = angle_obj

  @property
  def degrees(self):
    val = self._angle.degrees()
    return CallableFloat(val)

  def __call__(self):
    return self


class S2Point:
  """Adapter for S2Point."""

  def __init__(self, point_obj):
    self._point = point_obj


class S2LatLng:
  """Adapter for S2LatLng."""

  def __init__(self, lat_lng_obj):
    self._lat_lng = lat_lng_obj

  @classmethod
  def from_degrees(cls, lat, lon):
    return cls(_s2.S2LatLng.FromDegrees(lat, lon))

  @classmethod
  def from_point(cls, point):
    return cls(_s2.S2LatLng(point._point))

  def to_point(self):
    return S2Point(self._lat_lng.ToPoint())

  @property
  def lat(self):
    return S1Angle(self._lat_lng.lat())

  @property
  def lng(self):
    return S1Angle(self._lat_lng.lng())

  def __call__(self):
    return self


class S2CellId:
  """Adapter for S2CellId."""

  def __init__(self, cell_id_obj):
    self._cell_id = cell_id_obj

  @classmethod
  def from_token(cls, token):
    if isinstance(token, bytes):
      token = token.decode('utf-8')
    return cls(_s2.S2CellId.FromToken(token))

  def __hash__(self):
    return hash(self._cell_id.id())

  def __eq__(self, other):
    if isinstance(other, S2CellId):
      return self._cell_id.id() == other._cell_id.id()
    return False

  def id(self):
    return self._cell_id.id()


class S2Cell:
  """Adapter for S2Cell."""

  def __init__(self, cell_id):
    self._cell = _s2.S2Cell(cell_id._cell_id)

  def vertex(self, i):
    return S2Point(self._cell.GetVertex(i))


class S2Loop:

  def __init__(self, points):
    native_points = [p._point for p in points]
    self._loop = _s2.S2Loop(native_points)

  def normalize(self):
    self._loop.Normalize()


class S2Polygon:
  """Adapter for S2Polygon."""

  def __init__(self, polygon_obj=None):
    if polygon_obj is not None:
      self._polygon = polygon_obj
    else:
      self._polygon = _s2.S2Polygon()

  def init_nested(self, loops):
    native_loops = [l._loop for l in loops]
    self._polygon.InitNested(native_loops)

  @property
  def is_valid(self):
    return self._polygon.IsValid()

  @property
  def is_empty(self):
    return self._polygon.is_empty()

  def contains_point(self, point):
    return self._polygon.Contains(point._point)

  def area(self):
    return self._polygon.GetArea()


class S2CellUnion:
  """Adapter for S2CellUnion."""

  def __init__(self, cell_ids=None):
    if cell_ids is None:
      cell_ids = []

    if isinstance(cell_ids, S2CellUnion):
      self._union = cell_ids._union
    else:
      native_ids = [c._cell_id if hasattr(
          c, '_cell_id') else c for c in cell_ids]
      self._union = _s2.S2CellUnion(native_ids)

  def decode(self, decoder):
    return self._union.Decode(decoder._decoder)

  def encode(self, encoder):
    return self._union.Encode(encoder._encoder)

  def contains_point(self, point):
    return self._union.Contains(point._point)

  @property
  def cell_ids(self):
    return [S2CellId(c) for c in self._union.cell_ids()]

  def __iter__(self):
    return iter(self.cell_ids)

  def __len__(self):
    return len(self.cell_ids)

  def __getitem__(self, index):
    return self.cell_ids[index]


class S2RegionCoverer:
  """Adapter for S2RegionCoverer."""

  def __init__(self, min_level=None, max_level=None, max_cells=None):
    self._coverer = _s2.S2RegionCoverer()
    if min_level is not None:
      self._coverer.set_min_level(min_level)
    if max_level is not None:
      self._coverer.set_max_level(max_level)
    if max_cells is not None:
      self._coverer.set_max_cells(max_cells)

  def cover(self, region):
    return S2CellUnion(self._coverer.GetCovering(region._polygon))


class Encoder:

  def __init__(self):
    self._encoder = _s2.Encoder()

  def buffer(self):
    return self._encoder.buffer()


class Decoder:

  def __init__(self, data):
    self._decoder = _s2.Decoder(data)
