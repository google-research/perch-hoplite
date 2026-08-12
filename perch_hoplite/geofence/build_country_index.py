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

"""Creates a species-country geofencing table using threaded processing."""

from collections.abc import Sequence
import os
import tempfile

from absl import app
from absl import flags
from etils import epath
import geopandas as gpd
import pandas as pd
from perch_hoplite.geofence import indexing_lib

# --- CONFIGURATION ---
SIMPLIFY_TOLERANCE = 0.005
BASE_RANGEMAP_PATH = epath.Path("inat_ranges")
MAX_WORKERS = 16

_WORLD_DATA_PATH = flags.DEFINE_string(
    "world_data_path",
    None,
    "Path to world/administrative boundaries dataset. If None, "
    "uses Geopandas built-in naturalearth_lowres.",
)
_WORLD_DATA_COLUMN = flags.DEFINE_string(
    "world_data_column",
    None,
    "Column in the dataset to use for administrative regions. If None, "
    "automatically detects 'name' or 'ADMIN'.",
)
_OUTPUT_COLUMN_NAME = flags.DEFINE_string(
    "output_column_name",
    "countries",
    "Name of the output column in the Parquet file.",
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    "/tmp/species_country_index.parquet",
    "Path to write the resulting parquet index.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  print(os.cpu_count())
  print("Loading range maps...")
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = epath.Path(temp_dir)
    gpkg1_path = BASE_RANGEMAP_PATH / "iNaturalist_geomodel_Aves_1.gpkg"
    gpkg2_path = BASE_RANGEMAP_PATH / "iNaturalist_geomodel_Aves_2.gpkg"
    local_gpkg1_path = temp_path / gpkg1_path.name
    local_gpkg2_path = temp_path / gpkg2_path.name

    print(f"Copying {gpkg1_path} to {local_gpkg1_path}...")
    gpkg1_path.copy(local_gpkg1_path)
    print(f"Copying {gpkg2_path} to {local_gpkg2_path}...")
    gpkg2_path.copy(local_gpkg2_path)

    aves1 = gpd.read_file(local_gpkg1_path.as_posix())
    aves2 = gpd.read_file(local_gpkg2_path.as_posix())
    gdf = pd.concat([aves1, aves2], ignore_index=True)
    gdf = gdf.to_crs("EPSG:4326")

    indexing_lib.build_country_parquet_index(
        gdf,
        output_file=_OUTPUT_FILE.value,
        simplify_tolerance=SIMPLIFY_TOLERANCE,
        max_workers=MAX_WORKERS,
        world_data_path=_WORLD_DATA_PATH.value,
        world_data_column=_WORLD_DATA_COLUMN.value,
        output_column_name=_OUTPUT_COLUMN_NAME.value,
    )


if __name__ == "__main__":
  app.run(main)
