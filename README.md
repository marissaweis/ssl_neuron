# Self-supervised Representation Learning of Neuronal Morphologies

This repository contains code to the paper [Self-supervised Representation Learning of Neuronal Morphologies](https://arxiv.org/abs/2112.12482) by M.A. Weis, L. Pede, T. Lüddecke and A.S. Ecker (2021).

## Installation

```
python3 setup.py install
```

## Data

Download ABA data from [ABA](http://celltypes.brain-map.org/).

### Extract Data

Extract data using the [Allen Software Development Kit](http://alleninstitute.github.io/AllenSDK/cell_types.html). See [demo notebook](http://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html#Cell-Morphology-Reconstructions) on how to use the Allen Cell Types Database.


## Training
```
python3 ssl_neuron/main.py --config=configs/config.json
```

## Demos
For example usage of the code, see Jupyter notebooks in the "demos" folder.