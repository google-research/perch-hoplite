# Species Geofence Library

This library provides tools for converting species range maps (geospatial
polygons) into efficient S2 cell-based indices stored in Parquet format. This
allows for fast querying of species presence at a given latitude/longitude
coordinate by checking if the point falls within the S2 covering associated
with a species.

The core idea is to represent potentially complex species range polygons as a
collection of S2 cells. An `S2CellUnion` represents these cells and is
serialized as a binary buffer column in the Parquet file. If a query point falls
into any cell covered by a species' range, that species is considered present.

The library generates a single main artifact:

*   **Species S2 Cell Union Index**: A Parquet file mapping `species_id` to an
    encoded `S2CellUnion` binary representation containing the S2 covering.

## Files and Entry Points

*   **`build_inat_index_lib.py`**:
    *   `get_s2_covering(...)`: Takes a Shapely geometry object (Polygon or
        MultiPolygon) and returns an `s2.S2CellUnion` covering the geometry.
        `max_level` dictates the level of resolution (resolution of S2 cells) to
        use when computing the covering.
    *   `build_parquet_index(gdf, output_file, ...)`: Takes a GeoDataFrame
        containing species ranges, computes the S2 coverings in parallel, and
        saves the species IDs and encoded cell unions to a compressed Parquet
        index file.
*   **`build_inat_index.py`**:
    *   A script that directory copies and loads geopackage (`.gpkg`) files
        containing species range maps (such as from iNaturalist), filters and
        concatenates them, and runs `build_inat_index_lib.build_parquet_index` to
        build the species index.
*   **`inference.py`**:
    *   `GeofenceInference(index_path)`: A class that loads a Parquet index into
        memory for fast inference.
    *   `get_species_for_lat_lon(lat, lon)`: Returns a list of species names
        whose indexed ranges contain the S2 cell for the given lat/lon.

## Example Invocations

### 1. Building the Index

The script uses a threadpool to parallelize index creation, and is suited for
processing datasets up to a few thousand species. Place your geopackage range
maps (e.g., `iNaturalist_geomodel_Aves_1.gpkg` and
`iNaturalist_geomodel_Aves_2.gpkg`) in the `inat_ranges/` directory.

Run the build script:

```bash
python -m perch_hoplite.geofence.build_inat_index
```

This will run the processing to generate `/tmp/species_index.parquet` (or a
custom output path), which can then be loaded for inference.

### 2. Performing Inference

```python
from perch_hoplite.geofence import inference

# Load the index
infer = inference.GeofenceInference('/tmp/species_index.parquet')

# Query for species at a specific location
species_list = infer.get_species_for_lat_lon(lat=40.7, lon=-74.0)
print(species_list)
# ['SpeciesA', 'SpeciesB', ...]
```

## Testing Strategy

The library is tested using Python's `unittest` framework.

*   **`geofence_test.py`** contains the test suite:
    *   **Unit Tests**: Test core S2 conversion, handling of
        antimeridian-crossing geometries, and covering logic.
    *   **Inference Tests**: Verify `GeofenceInference` query behavior against
        small, dynamically generated indices.
    *   **End-to-End Test**: `test_build_and_inference_e2e` generates a
        temporary Parquet index from test species GeoDataFrames and queries it
        to confirm that presence/absence checks are correct across overlapping
        and non-overlapping ranges.

To run the geofence tests:

```bash
python -m unittest perch_hoplite.geofence.geofence_test
```

## Acknowledgment

Range map data is derived from the iNaturalist Open Range Map Dataset.

**Citation:**
iNaturalist. (2026). iNaturalist Open Range Map Dataset 2.32.
Available at: https://www.inaturalist.org. Accessed July 7, 2026.
