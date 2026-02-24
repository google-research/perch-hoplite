# Agile Modeling with Perch-Hoplite

This directory contains tools for Agile bird song modeling with Perch-Hoplite.
These tools are intended to support embedding large audio datasets, adding
labels and metadata, and training and evaluating audio classifiers.

## Data Organization

The embedding pipeline assumes that audio files are organized into directories,
where each top-level directory within the `base_path` represents a
**deployment**. For example, with a directory structure like:

```
my_dataset/
├── deployment_A/
│   ├── recording01.wav
│   └── recording02.wav
├── deployment_B/
│   └── recording03.wav
...
```

`deployment_A` and `deployment_B` will be treated as deployment names.

**Recordings** are identified by their relative path from the `base_path`,
including the deployment directory (e.g., `deployment_A/recording01.wav`).
This relative path serves as the `file_id` for recordings when linking
metadata or annotations.

## Adding metadata to the Hoplite Database

The Agile embedding pipeline supports adding metadata to deployments and
recordings in the Hoplite database. Metadata is loaded from CSV files
located in the `base_path` of each `AudioSourceConfig`.

To add metadata, create the following three files in the root of your dataset
directory:

1.  **`metadata_description.csv`**: This file describes the metadata fields you
    want to add. It should contain the following columns:
    *   `field_name`: The name of the metadata field (e.g., `habitat`).
    *   `metadata_level`: The level at which metadata applies, either
        `deployment` or `recording`.
    *   `type`: The data type of the field. Supported types are `str`, `float`,
        `int`, and `bytes`.
    *   `description`: An optional description of the field.

2.  **`deployments_metadata.csv`**: This file contains metadata for each
    deployment. The first column must be the deployment identifier (which
    corresponds to the directory name if audio files are in
    `deployment/recording.wav`
    format), and subsequent columns should match `field_name`s from
    `metadata_description.csv` where `metadata_level` is `deployment`.

3.  **`recordings_metadata.csv`**: This file contains metadata for each
    recording. The first column must be the recording identifier (e.g.
    `deployment/recording.wav`),
    and subsequent columns should match `field_name`s from
    `metadata_description.csv` where `metadata_level` is `recording`.

### Example

**`metadata_description.csv`**

```csv
field_name,metadata_level,type,description
deployment_name,deployment,str,Deployment identifier.
habitat,deployment,str,Habitat type.
latitude,deployment,float,Deployment latitude.
file_id,recording,str,Recording identifier.
mic_type,recording,str,Microphone type.
```

**`deployments_metadata.csv`**

```csv
deployment_name,habitat,latitude
DEP01,"forest",47.6
DEP02,"grassland",45.1
```

**`recordings_metadata.csv`**

```csv
file_id,mic_type
DEP01/rec001.wav,"MicA"
DEP01/rec002.wav,"MicB"
DEP02/rec001.wav,"MicA"
```

When `EmbedWorker.process_all()` is run, it will detect these files, load the
metadata, and insert it into the database alongside new deployments and
recordings. Metadata fields can then be accessed as attributes on `Deployment`
and `Recording` objects returned by the database interface (e.g.,
`deployment.habitat`, `recording.mic_type`).

## Adding Annotations

If you have existing annotations for your audio data, Hoplite can ingest these
during the embedding process. Annotations should be stored in CSV files named
`annotations.csv` alongside your audio data. Each `annotations.csv` should
contain columns for `recording` (the filename or file_id of the audio),
`start_offset_s`, `end_offset_s`, `label`, and `label_type` ('positive',
'negative', or 'uncertain'). When embeddings are generated, Hoplite will find
any relevant annotations and add them to the database, associating them with the
appropriate time windows.

### Example

**`annotations.csv`**

```csv
recording,start_offset_s,end_offset_s,label,label_type
DEP01/rec001.wav,10.0,15.0,MyBird,positive
DEP01/rec001.wav,20.0,25.0,OtherBird,negative
DEP02/rec001.wav,5.0,10.0,MyBird,positive
```

## Colab Notebooks

This directory includes Colab notebooks to guide users through embedding audio,
adding annotations, and training agile classifiers.

These notebooks are designed for use in Google Colab and make use of interactive
forms (e.g., dropdowns and text fields) via cell parameters (`#@param`).
While developed for Colab, the notebooks are also compatible with standard
Jupyter environments, although the interactive form elements will not be
rendered.

The notebooks provided are:

*   **`01_embed_audio.ipynb`**: This notebook guides you through the process of
    embedding audio files from a dataset using a specified pre-trained model
    (e.g., Perch v2, BirdNet) and saving them into a Hoplite database. It
    handles dataset configuration, database initialization, and running the
    embedding process.
*   **`02_agile_modeling.ipynb`**: This notebook focuses on the interactive
    modeling process. It allows you to search the embedding database using
    example audio, display search results, label data as positive or negative,
    and then train or retrain a simple linear classifier based on these labels.
    You can also use the trained classifier to run inference or perform
    margin-based sampling to find examples for further annotation.
*   **`03_call_density.ipynb`**: This notebook shows how to use Hoplite to
    compute aggregate call density statistics, which can act as an indicator
    of species abundance in many cases.
    (As described in: https://arxiv.org/abs/2402.15360)
*   **`99_migrate_db.ipynb`**: A utility notebook for migrating Hoplite
    databases created with `perch-hoplite < 1.0` to the format used by
    `perch-hoplite >= 1.0`.
