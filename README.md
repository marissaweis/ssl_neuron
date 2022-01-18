# Self-supervised Representation Learning of Neuronal Morphologies

This repository contains code to the paper [Self-supervised Representation Learning of Neuronal Morphologies](https://arxiv.org/abs/2112.12482) by M.A. Weis, L. Pede, T. Lüddecke and A.S. Ecker (2021).

## Installation

```
python3 setup.py install
```

## Data

Download data from [ABA](http://celltypes.brain-map.org/).

### Extract Data

Extract data using [AllenSDK](https://allensdk.readthedocs.io/en/latest/).


## Training
```
python3 ssl_neuron/main.py --config=configs/config.json
```

## Demos
For example usage of the code, see Jupyter notebooks in the "demos" folder.