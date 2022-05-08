## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install requirements.txt
```
SimCLRv2 extractor requires a path to the directory where the model is stored specified in the feature_extractor.py file.

## Usage
Data needs to be in PNG format. The map file contains a pickled list of patients. Each patient is represented by a list of its PNG filenames.

```bash
python anonymity.vpi_block_subset
    --slide_dir {data_directory}
    --map_path {pickled_map_path}
    --extractor {extractor}
    --metric {metric}
```
