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

"""Creates a species geofencing table using threaded processing."""

from collections.abc import Sequence
import os
import tempfile

from absl import app
from etils import epath
import geopandas as gpd
import pandas as pd
from perch_hoplite.geofence import indexing_lib

# --- CONFIGURATION ---
MIN_LEVEL = 3
MAX_LEVEL = 9
SIMPLIFY_TOLERANCE = 0.005  # ~500m. Increase this if recursion is still slow.
BASE_RANGEMAP_PATH = epath.Path("inat_ranges")
MAX_WORKERS = 16  # os.cpu_count()
DEBUG_SPECIES = ""


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  print(os.cpu_count())
  print("Loading range maps...")
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = epath.Path(temp_dir)
    gpkgs = [
        "iNaturalist_geomodel_Aves_1.gpkg",
        "iNaturalist_geomodel_Aves_2.gpkg",
        "iNaturalist_geomodel_Mammalia.gpkg",
        "iNaturalist_geomodel_Amphibia.gpkg",
    ]
    gdf_list = []
    for gpkg in gpkgs:
      gpkg_path = BASE_RANGEMAP_PATH / gpkg
      local_gpkg_path = temp_path / gpkg_path.name
      print(f"Copying {gpkg_path} to {local_gpkg_path}...")
      gpkg_path.copy(local_gpkg_path)
      gdf = gpd.read_file(local_gpkg_path.as_posix())
      gdf_list.append(gdf)

    gdf = pd.concat(gdf_list, ignore_index=True)
    gdf = gdf.to_crs("EPSG:4326")
    if DEBUG_SPECIES:
      gdf = gdf[gdf["name"] == DEBUG_SPECIES]
      print(f"Filtering for debug species: {DEBUG_SPECIES}")
    indexing_lib.build_parquet_index(
        gdf,
        min_level=MIN_LEVEL,
        max_level=MAX_LEVEL,
        simplify_tolerance=SIMPLIFY_TOLERANCE,
        max_workers=MAX_WORKERS,
        output_file="/tmp/species_index.parquet",
    )


if __name__ == "__main__":
  app.run(main)
