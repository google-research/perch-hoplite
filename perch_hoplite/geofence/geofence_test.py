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

import os
import tempfile

import geopandas as gpd
from geopandas import datasets as gpd_datasets
from perch_hoplite.geofence import indexing_lib
from perch_hoplite.geofence import inference
from perch_hoplite.geofence import s2adapter as s2
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import geometry
from shapely import wkt

from absl.testing import absltest
from absl.testing import parameterized

_GALLIRALLUS_SYLVESTRIS_WKT = (
    "POLYGON ((159.45915746698597 -31.723197702409866, 159.47146394460236"
    " -31.466996517035447, 159.21821112249597 -31.32868461554112,"
    " 158.951847354469 -31.445992751870005, 158.93798026988168"
    " -31.702274892190808, 159.19203734033817 -31.841170918604426,"
    " 159.45915746698597 -31.723197702409866))"
)
_DREPANIS_COCCINEA_WKT = """MULTIPOLYGON (((-157.87948726890173 21.38133798936681,
    -157.71903059925305 21.58591531077887,
    -157.82764266367306 21.81389575390711,
    -158.09687331066377 21.8370772231161,
    -158.2569306849044 21.632471864251052,
    -158.14816006623352 21.404713500139515,
    -157.87948726890173 21.38133798936681)),
    ((-154.70956029588518 19.524365273797507,
    -154.81517157130796 19.757710106524527,
    -155.08415677687768 19.78680441509366,
    -155.19017238344955 20.01955711803261,
    -155.45926966595988 20.047962326143995,
    -155.56568151314605 20.280099736606918,
    -155.8348678112481 20.307800737053935,
    -155.9971011948672 20.103484245851416,
    -155.89046267131891 19.87154338949282,
    -156.0523185194713 19.66692485010155,
    -155.94584108755356 19.43456689532171,
    -155.67772859171328 19.40662400039649,
    -155.57163357626203 19.173664141060723,
    -155.30360251801108 19.145042630376675,
    -155.141137481466 19.349490114533737,
    -154.87268225188404 19.320301026357814,
    -154.70956029588518 19.524365273797507)),
    ((-156.5329973021786 21.25763322777263,
    -156.69494473036684 21.053526039894017,
    -156.9642586449114 21.0790277959985,
    -157.12547012527222 20.87453073094006,
    -157.01772644482568 20.644584548877553,
    -156.74896091360347 20.6189249171094,
    -156.64157708467906 20.388291838083376,
    -156.37282733455194 20.36189745623896,
    -156.21091981341095 20.566282136295595,
    -155.94166767307604 20.539300054125704,
    -155.77905534121868 20.743289028357545,
    -155.8860167862531 20.974322655470893,
    -156.15581201091368 21.001172961649512,
    -156.26315591266606 21.231528538035565,
    -156.5329973021786 21.25763322777263)),
    ((-159.7069393678917 21.966434215807634,
    -159.4392851171177 21.946031821419872,
    -159.28089381424 22.151018736816393,
    -159.39061017509593 22.37640770967177,
    -159.65883092501284 22.39656802869345,
    -159.81676550614728 22.191581067749187,
    -160.08424573554342 22.211163589500465,
    -160.24132981720263 22.005854264468137,
    -160.13140753325632 21.780959049517598,
    -159.86449403779105 21.761121160481782,
    -159.7069393678917 21.966434215807634)))"""
_HEMIGNATHUS_WILSONI_WKT = """MULTIPOLYGON (((-155.45926966595988 20.047962326143995,
    -155.6218140922834 19.843721988739976,
    -155.89046267131891 19.87154338949282,
    -156.0523185194713 19.66692485010155,
    -155.94584108755356 19.43456689532171,
    -155.67772859171328 19.40662400039649,
    -155.57163357626203 19.173664141060723,
    -155.30360251801108 19.145042630376675,
    -155.141137481466 19.349490114533737,
    -155.24699740007287 19.58264816891696,
    -155.08415677687768 19.78680441509366,
    -155.19017238344955 20.01955711803261,
    -155.45926966595988 20.047962326143995)))"""


class GeofenceTest(parameterized.TestCase):

  def _s2_cell_to_shapely_poly(self, cell_id):
    cell = s2.S2Cell(cell_id)
    vertices = []
    for i in range(4):
      p = s2.S2LatLng.from_point(cell.vertex(i))
      vertices.append((p.lng.degrees, p.lat.degrees))
    return geometry.Polygon(vertices)

  def _get_test_geometry(self, species_name: str) -> geometry.base.BaseGeometry:
    if species_name == "Gallirallus sylvestris":
      return wkt.loads(_GALLIRALLUS_SYLVESTRIS_WKT)
    elif species_name == "Drepanis coccinea":
      return wkt.loads(_DREPANIS_COCCINEA_WKT)
    elif species_name == "Hemignathus wilsoni":
      return wkt.loads(_HEMIGNATHUS_WILSONI_WKT)
    else:
      raise ValueError(f"Unknown species: {species_name}")

  def _get_test_gdf(self) -> gpd.GeoDataFrame:
    species1 = "Gallirallus sylvestris"
    species2 = "Drepanis coccinea"
    species3 = "Hemignathus wilsoni"
    geom1 = self._get_test_geometry(species1)
    geom2 = self._get_test_geometry(species2)
    geom3 = self._get_test_geometry(species3)
    gdf = gpd.GeoDataFrame(
        [
            {"name": species1, "geometry": geom1},
            {"name": species2, "geometry": geom2},
            {"name": species3, "geometry": geom3},
        ],
        crs="EPSG:4326",
    )
    return gdf

  def test_shapely_polygon_to_s2_polygon(self):
    shape = geometry.box(-0.01, -0.01, 0.01, 0.01)
    s2_poly = indexing_lib._shapely_polygon_to_s2_polygon(shape)
    self.assertIsInstance(s2_poly, s2.S2Polygon)
    self.assertTrue(s2_poly.is_valid)
    self.assertFalse(s2_poly.is_empty)
    self.assertGreater(s2_poly.area(), 0)
    center = s2.S2LatLng.from_degrees(0, 0).to_point()
    self.assertTrue(s2_poly.contains_point(center))

  def test_shapely_polygon_to_s2_polygon_antimeridian(self):
    # Polygon crossing antimeridian.
    shape = geometry.Polygon(
        [(179, -1), (179, 1), (-179, 1), (-179, -1), (179, -1)]
    )
    s2_poly = indexing_lib._shapely_polygon_to_s2_polygon(shape)
    self.assertIsInstance(s2_poly, s2.S2Polygon)
    self.assertTrue(s2_poly.is_valid)
    self.assertFalse(s2_poly.is_empty)
    self.assertGreater(s2_poly.area(), 0)
    # Check that point 180,0 is contained.
    center = s2.S2LatLng.from_degrees(0, 180).to_point()
    self.assertTrue(s2_poly.contains_point(center))

  def test_get_s2_covering_max_level(self):
    cell_id = s2.S2CellId.from_token(b"89c259b")  # level 13
    shape = self._s2_cell_to_shapely_poly(cell_id)
    max_level = 13
    cell_union = indexing_lib.get_s2_covering(
        shape, min_level=3, max_level=max_level
    )
    self.assertLess(len(cell_union.cell_ids), 50)

  def test_get_s2_covering_recursion(self):
    cell_id = s2.S2CellId.from_token(b"89c259")  # level 12
    shape = self._s2_cell_to_shapely_poly(cell_id)
    max_level = 13
    cell_union = indexing_lib.get_s2_covering(
        shape, min_level=3, max_level=max_level, simplify_tolerance=0.0
    )
    self.assertLess(len(cell_union.cell_ids), 50)

  def test_get_s2_covering_simple_polygon(self):
    # A small box around lat/lng 0,0.
    shape = geometry.box(-0.01, -0.01, 0.01, 0.01)
    max_level = 7
    cell_union = indexing_lib.get_s2_covering(
        shape, min_level=3, max_level=max_level
    )
    self.assertNotEmpty(cell_union.cell_ids)
    self.assertLess(len(cell_union.cell_ids), 500)

  def test_get_s2_covering_empty_geometry(self):
    poly = geometry.Polygon()
    cell_union = indexing_lib.get_s2_covering(poly)
    self.assertEmpty(cell_union.cell_ids)

  def test_get_s2_covering_gallirallus_sylvestris(self):
    # This is a polygon for Gallirallus sylvestris, which caused problems
    # due to the antimeridian.
    shape = self._get_test_geometry("Gallirallus sylvestris")
    cell_union = indexing_lib.get_s2_covering(
        shape, simplify_tolerance=0.0, max_level=12
    )
    self.assertNotEmpty(cell_union.cell_ids)

  def test_inference_gallirallus_sylvestris(self):
    species_name = "Gallirallus sylvestris"
    shape = self._get_test_geometry(species_name)
    cell_union = indexing_lib.get_s2_covering(
        shape, simplify_tolerance=0.0, max_level=12
    )
    encoder = s2.Encoder()
    cell_union.encode(encoder)
    table = pa.table({
        "species_id": [species_name],
        "cell_union": pa.array([encoder.buffer()], type=pa.binary()),
    })
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    temp_file_path = os.path.join(temp_dir.name, "test_index.parquet")
    pq.write_table(table, temp_file_path)

    infer = inference.GeofenceInference(temp_file_path)

    # Point known to be inside Lord Howe Island.
    species_in = infer.get_species_for_lat_lon(lat=-31.55, lon=159.08)
    self.assertIn(species_name, species_in)

    # Point known to be outside.
    species_out = infer.get_species_for_lat_lon(lat=0.0, lon=0.0)
    self.assertNotIn(species_name, species_out)

  def test_inference_drepanis_coccinea(self):
    species_name = "Drepanis coccinea"
    shape = self._get_test_geometry(species_name)
    cell_union = indexing_lib.get_s2_covering(
        shape, simplify_tolerance=0.0, max_level=12
    )
    encoder = s2.Encoder()
    cell_union.encode(encoder)
    table = pa.table({
        "species_id": [species_name],
        "cell_union": pa.array([encoder.buffer()], type=pa.binary()),
    })
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    temp_file_path = os.path.join(temp_dir.name, "test_index_multi.parquet")
    pq.write_table(table, temp_file_path)

    infer = inference.GeofenceInference(temp_file_path)

    # Point in Hawaii.
    species_in = infer.get_species_for_lat_lon(lat=20.8, lon=-156.3)
    self.assertIn(species_name, species_in)

    # Point known to be outside.
    species_out = infer.get_species_for_lat_lon(lat=0.0, lon=0.0)
    self.assertNotIn(species_name, species_out)

  def test_build_parquet_index(self):
    gdf = self._get_test_gdf()
    species1 = "Gallirallus sylvestris"
    species2 = "Drepanis coccinea"
    species3 = "Hemignathus wilsoni"
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    temp_file_path = os.path.join(temp_dir.name, "test_build_index.parquet")
    indexing_lib.build_parquet_index(
        gdf,
        output_file=temp_file_path,
        max_level=12,
        simplify_tolerance=0.0,
    )

    ranges_path = temp_file_path
    self.assertTrue(os.path.exists(ranges_path))

    ranges_table = pq.read_table(ranges_path)
    self.assertNotEmpty(ranges_table)
    self.assertIn(species1, ranges_table["species_id"].to_pylist())
    self.assertIn(species2, ranges_table["species_id"].to_pylist())
    self.assertIn(species3, ranges_table["species_id"].to_pylist())
    self.assertIsNotNone(ranges_table["cell_union"])

  def test_build_and_inference_e2e(self):
    gdf = self._get_test_gdf()
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    temp_file_path = os.path.join(temp_dir.name, "e2e_index.parquet")
    indexing_lib.build_parquet_index(
        gdf,
        output_file=temp_file_path,
        max_level=12,
        simplify_tolerance=0.0,
    )
    infer = inference.GeofenceInference(temp_file_path)

    # Test point for Gallirallus sylvestris on Lord Howe Island.
    species_in_gallirallus = infer.get_species_for_lat_lon(
        lat=-31.55, lon=159.08
    )
    self.assertIn("Gallirallus sylvestris", species_in_gallirallus)
    self.assertNotIn("Drepanis coccinea", species_in_gallirallus)
    self.assertNotIn("Hemignathus wilsoni", species_in_gallirallus)

    # Test point for Drepanis coccinea on Maui.
    # The I'iwi (D. coccinea) should be present, but not Aki (H. wilsoni).
    species_in_drepanis = infer.get_species_for_lat_lon(lat=20.8, lon=-156.3)
    self.assertIn("Drepanis coccinea", species_in_drepanis)
    self.assertNotIn("Hemignathus wilsoni", species_in_drepanis)

    # Point known to be outside both.
    species_out = infer.get_species_for_lat_lon(lat=0.0, lon=0.0)
    self.assertNotIn("Gallirallus sylvestris", species_out)
    self.assertNotIn("Drepanis coccinea", species_out)

    # Test point for Hawaii Big Island, where both D. coccinea and
    # H. wilsoni should be present.
    species_in_big_island = infer.get_species_for_lat_lon(lat=19.7, lon=-155.5)
    self.assertIn("Drepanis coccinea", species_in_big_island)
    self.assertIn("Hemignathus wilsoni", species_in_big_island)

  def test_simple_inference(self):
    # Load the default index.
    infer = inference.GeofenceInference(index_path=inference.INAT_INDEX_PATH)
    species_in = infer.get_species_for_lat_lon(lat=-31.55, lon=159.08)
    self.assertIn("Gallirallus sylvestris", species_in)

  def test_inference_default_country_index(self):
    infer = inference.GeofenceInference()
    self.assertLen(
        infer.get_species_for_country("United States of America"), 1173
    )


if __name__ == "__main__":
  absltest.main()
