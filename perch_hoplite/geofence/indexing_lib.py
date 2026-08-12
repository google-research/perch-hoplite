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

"""Library for creating species geofencing tables."""

import concurrent.futures
import time

import geopandas as gpd
from geopandas import datasets as gpd_datasets
from perch_hoplite.geofence import s2adapter as s2
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

# Skip species with errors in the geometry.
SKIP_SPECIES = [
    "Columba livia",
    "Pygoscelis adeliae",
    "Aptenodytes forsteri",
    "Calidris canutus",
    "Calidris alpina",
    "Oceanites oceanicus",
    "Pagodroma nivea",
    "Thalassoica antarctica",
    "Larus hyperboreus",
    "Larus fuscus",
    "Sterna paradisaea",
    "Stercorarius parasiticus",
    "Rissa tridactyla",
    "Xema sabini",
    "Gavia stellata",
    "Falco peregrinus",
    "Pluvialis squatarola",
    "Mergus serrator",
    "Branta bernicla",
    "Clangula hyemalis",
    "Asio flammeus",
    "Phalaropus fulicarius",
    "Calcarius lapponicus",
    "Stercorarius maccormicki",
]


def _shapely_point_to_s2_point(p):
  return s2.S2LatLng.from_degrees(p[1], p[0]).to_point()


def _shapely_ring_to_s2_loop(ring):
  """Converts a shapely LinearRing to an S2Loop."""
  points = [_shapely_point_to_s2_point(p) for p in ring.coords[:-1]]
  if len(points) < 3:
    return None
  loop = s2.S2Loop(points)
  # s2 loops must be oriented CCW.
  # loop.normalize() ensures CCW orientation.
  loop.normalize()
  return loop


def _shapely_polygon_to_s2_polygon(poly):
  """Converts a shapely Polygon to an S2Polygon."""
  s2poly = s2.S2Polygon()
  loops = []
  try:
    exterior = _shapely_ring_to_s2_loop(poly.exterior)
  except Exception:  # pylint: disable=broad-except
    return None
  if exterior is None:
    return None
  loops.append(exterior)
  for interior in poly.interiors:
    try:
      interior_loop = _shapely_ring_to_s2_loop(interior)
      if interior_loop is not None:
        loops.append(interior_loop)
    except Exception:  # pylint: disable=broad-except
      # Skip invalid interior loops.
      continue
  s2poly.init_nested(loops)
  return s2poly


def _make_region_coverer(
    min_level: int,
    max_level: int,
    max_cells: int,
) -> s2.S2RegionCoverer:
  """Returns an S2RegionCoverer with the given options."""
  return s2.S2RegionCoverer(
      min_level=min_level, max_level=max_level, max_cells=max_cells
  )


def get_s2_covering(
    geometry_obj,
    min_level: int = 2,
    max_level: int = 9,
    max_cells: int = 512,
    simplify_tolerance: float = 0.005,
) -> s2.S2CellUnion:
  """Returns a list of (id_start, id_end) tuples for any Shapely geometry."""
  # 0. Fix invalid geometries.
  geometry_obj = geometry_obj.buffer(0)
  if geometry_obj.is_empty:
    return s2.S2CellUnion([])
  # 1. Simplify to skip micro-details (biological ranges are rarely cm-precise)
  simplified_geom = geometry_obj.simplify(
      simplify_tolerance, preserve_topology=True
  )
  geom = simplified_geom
  if simplified_geom.is_empty and not geometry_obj.is_empty:
    geom = geometry_obj

  coverer = _make_region_coverer(min_level, max_level, max_cells)

  if geom.geom_type == "Polygon":
    polygons = [geom]
  elif geom.geom_type == "MultiPolygon":
    polygons = list(geom.geoms)
  else:
    print(f"Unsupported geometry type: {geom.geom_type}")
    return s2.S2CellUnion([])

  all_cell_ids = set()
  for poly in polygons:
    if poly.is_empty:
      continue
    s2poly = _shapely_polygon_to_s2_polygon(poly)
    if s2poly is None or s2poly.is_empty:
      continue
    covering = coverer.cover(s2poly)
    all_cell_ids.update(covering)
  return s2.S2CellUnion(list(all_cell_ids))


def build_parquet_index(
    gdf: gpd.GeoDataFrame,
    output_file: str | None = "/tmp/species_index.parquet",
    min_level: int = 2,
    max_level: int = 9,
    max_cells: int = 512,
    simplify_tolerance: float = 0.005,
    max_workers: int = 16,
) -> pa.Table:
  """Processes the GeoDataFrame and saves it to a compressed Parquet file.

  Args:
    gdf: GeoDataFrame with geometry column containing species ranges. Expected
      to be in EPSG:4326.
    output_file: If not None, path to write the resulting parquet index.
    min_level: Minimum S2 level for cells in covering.
    max_level: Maximum S2 level for cells in covering.
    max_cells: Maximum number of cells in covering.
    simplify_tolerance: Tolerance for geometry simplification.
    max_workers: Number of threads for concurrent processing.

  Returns:
    Arrow table containing species_id and encoded S2CellUnion.
  """
  species_unions: list[tuple[str, s2.S2CellUnion]] = []

  def process_row(args):
    """Get s2 covering for one row."""
    idx, row = args
    name = row.get("name", f"sp_{idx}")
    if name in SKIP_SPECIES:
      print(f"Skipping {name}...")
      return name, s2.S2CellUnion([])
    try:
      if row["geometry"].is_empty:
        print(f"Skipping {name} due to empty geometry...")
        return name, s2.S2CellUnion([])
      start_time = time.time()
      cell_union = get_s2_covering(
          row["geometry"],
          min_level=min_level,
          max_level=max_level,
          max_cells=max_cells,
          simplify_tolerance=simplify_tolerance,
      )
      elapsed = time.time() - start_time
      if not cell_union.cell_ids and not row["geometry"].is_empty:
        print(f"!!! {name} resulted in 0 cells !!!")
      print(
          f"completed {idx} {name} with {len(cell_union.cell_ids)} cells in"
          f" {elapsed:.2f}s"
      )
      return name, cell_union
    except Exception as e:  # pylint: disable=broad-except
      print(f"\nError processing {name}: {e}")
      return name, s2.S2CellUnion([])

  with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
    results = tqdm.tqdm(
        executor.map(process_row, gdf.iterrows()),
        total=len(gdf),
        desc="Processing species",
    )
    for name, cell_union in results:
      species_unions.append((name, cell_union))

  all_names = []
  all_encoded_unions = []
  for name, cell_union in species_unions:
    if not cell_union.cell_ids:
      continue
    encoder = s2.Encoder()
    cell_union.encode(encoder)
    all_names.append(name)
    all_encoded_unions.append(encoder.buffer())

  table = pa.table({
      "species_id": all_names,
      "cell_union": pa.array(all_encoded_unions, type=pa.binary()),
  })

  if output_file:
    pq.write_table(table, output_file, compression="snappy")
  return table


def build_country_parquet_index(
    gdf: gpd.GeoDataFrame,
    output_file: str | None = "/tmp/species_country_index.parquet",
    simplify_tolerance: float = 0.005,
    max_workers: int = 16,  # pylint: disable=unused-argument
    world_data_path: str | None = None,
    world_data_column: str | None = None,
    output_column_name: str = "countries",
) -> pa.Table:
  """Processes the GeoDataFrame and saves it to a compressed Parquet file mapping species to admin regions.

  Args:
    gdf: GeoDataFrame with geometry column containing species ranges. Expected
      to be in EPSG:4326.
    output_file: If not None, path to write the resulting parquet index.
    simplify_tolerance: Tolerance for geometry simplification.
    max_workers: Unused, kept for API compatibility.
    world_data_path: Path to the world country borders dataset. If None,
      defaults to the geopandas built-in "naturalearth_lowres" dataset.
    world_data_column: Column name in the administrative regions dataset to use
      for grouping. If None, automatically detects "name" or "ADMIN".
    output_column_name: Name of the output column in the Parquet file.

  Returns:
    Arrow table containing species_id and output_column_name.
  """
  if world_data_path is None:
    world_data_path = gpd_datasets.get_path("naturalearth_lowres")
  world = gpd.read_file(world_data_path)

  if world_data_column is None:
    for col in ("name", "ADMIN"):
      if col in world.columns:
        world_data_column = col
        break
    else:
      raise ValueError(
          "Could not detect column for admin names. Please specify"
          " world_data_column."
      )
  elif world_data_column not in world.columns:
    raise ValueError(
        f"Column '{world_data_column}' not found in world dataset columns: "
        f"{list(world.columns)}"
    )

  # Keep only the admin column of interest and the geometry to prevent sjoin
  # conflicts
  world = world[[world_data_column, "geometry"]]

  # Rename world_data_column to a unique name to avoid sjoin suffix conflicts
  join_column = "_admin_region_name"
  world = world.rename(columns={world_data_column: join_column})

  world = world.to_crs("EPSG:4326")

  # Drop species that have errors
  working_gdf = gdf[~gdf.get("name", gdf.index).isin(SKIP_SPECIES)].copy()

  # Fix geometries and remove empty ones
  working_gdf["geometry"] = working_gdf.geometry.buffer(0)
  if simplify_tolerance > 0:
    simplified = working_gdf.geometry.simplify(
        simplify_tolerance, preserve_topology=True
    )
    empty_mask = simplified.is_empty & ~working_gdf.geometry.is_empty
    working_gdf["geometry"] = simplified.mask(empty_mask, working_gdf.geometry)

  working_gdf = working_gdf[~working_gdf.geometry.is_empty]

  print("Joining species ranges with administrative borders...")
  # Spatial join species ranges against admin regions
  joined = gpd.sjoin(working_gdf, world, how="inner", predicate="intersects")

  # Group by species and get unique lists of admin regions
  species_regions = (
      joined.groupby("name")[join_column].unique().apply(list).reset_index()
  )

  table = pa.table({
      "species_id": species_regions["name"].tolist(),
      output_column_name: pa.array(species_regions[join_column].tolist()),
  })

  if output_file:
    pq.write_table(table, output_file, compression="snappy")
  return table
