# SSFNR
- Subspace-Frequency Regularization for Hyperspectral Image Super-Resolution

## Prepare dataset
- Datasets can be downloaded from the [CAVE] (https://www.cs.columbia.edu/CAVE/databases/multispectral/). After downloading, please **place the files according to the following directory structure**:

/data/CAVE/complete_ms_data/
├── balloons_ms
├── beads_ms
├── ...
├── watercolors_ms

## Data preprocessing
- For the first run, edit the `option.py` file and set: prepare = "Yes".

## To train:
```python
python main.py
```
