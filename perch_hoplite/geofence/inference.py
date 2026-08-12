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

"""Inference for species geofencing."""

from etils import epath
import pandas as pd
from perch_hoplite import path_utils
from perch_hoplite.geofence import s2adapter as s2

INAT_INDEX_PATH = path_utils.get_absolute_path("geofence/species_index.parquet")
COUNTRY_INDEX_PATH = path_utils.get_absolute_path(
    "geofence/species_country_index.parquet"
)


class GeofenceInference:
  """Loads a geofence index and returns species for a given lat/lon."""

  def __init__(
      self,
      index_path: epath.Path | str | None = INAT_INDEX_PATH,
      country_index_path: epath.Path | str | None = COUNTRY_INDEX_PATH,
      regions_column: str | None = None,
  ):
    """Initializes the GeofenceInference.

    Args:
      index_path: Path to the Parquet file containing the geofence index.
      country_index_path: Path to the Parquet file containing the country index.
      regions_column: Name of the column containing region names in the country
        index file. If None, automatically detects "countries" or the first
        non-species_id column.
    """
    self.index_path = epath.Path(index_path) if index_path else None
    self.country_index_path = (
        epath.Path(country_index_path) if country_index_path else None
    )

    self._species_unions: list[tuple[str, s2.S2CellUnion]] = []
    if self.index_path and self.index_path.exists():
      with self.index_path.open("rb") as f:
        df = pd.read_parquet(f)
      for _, row in df.iterrows():
        decoder = s2.Decoder(row["cell_union"])
        cell_union = s2.S2CellUnion([])
        if cell_union.decode(decoder):
          self._species_unions.append((row["species_id"], cell_union))
        else:
          print(f"Failed to decode S2CellUnion for {row['species_id']}")

    self._country_species: dict[str, list[str]] = {}
    if self.country_index_path and self.country_index_path.exists():
      with self.country_index_path.open("rb") as f:
        country_df = pd.read_parquet(f)

      if regions_column is None:
        if "countries" in country_df.columns:
          regions_column = "countries"
        else:
          other_cols = [c for c in country_df.columns if c != "species_id"]
          if other_cols:
            regions_column = other_cols[0]
          else:
            regions_column = "countries"

      for _, row in country_df.iterrows():
        species_id = row["species_id"]
        regions = row.get(regions_column, [])
        if regions is not None:
          for region in regions:
            if region not in self._country_species:
              self._country_species[region] = []
            self._country_species[region].append(species_id)

  def get_species_for_lat_lon(self, lat: float, lon: float) -> list[str]:
    """Returns a list of species whose ranges contain the given lat/lon.

    Args:
      lat: Latitude of the point.
      lon: Longitude of the point.

    Returns:
      A list of species names.
    """
    point = s2.S2LatLng.from_degrees(lat, lon).to_point()
    result = []
    for species_id, cell_union in self._species_unions:
      if cell_union.contains_point(point):
        result.append(species_id)
    return result

  def get_species_for_region(self, region: str) -> list[str]:
    """Returns a list of species whose ranges intersect the given region.

    Args:
      region: Name of the region.

    Returns:
      A list of species names.
    """
    return self._country_species.get(region, [])

  def get_species_for_country(self, country: str) -> list[str]:
    """Returns a list of species whose ranges intersect the given country.

    Args:
      country: Name of the country.

    Returns:
      A list of species names.
    """
    return self.get_species_for_region(country)
