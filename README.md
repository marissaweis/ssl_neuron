# Self-supervised Representation Learning for Neuronal Morphologies

This repository contains code to the paper [Self-Supervised Graph Representation Learning for Neuronal Morphologies](https://openreview.net/forum?id=ThhMzfrd6r) by M.A. Weis, L. Hansel, T. Lüddecke and A.S. Ecker (2023).

## Installation

```
python3 setup.py install
```

## Data

Extract data using the [Allen Software Development Kit](http://alleninstitute.github.io/AllenSDK/cell_types.html). See [demo notebook](http://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html#Cell-Morphology-Reconstructions) on how to use the Allen Cell Types Database.

See [extract_allen_data.ipynb](https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/data/extract_allen_data.ipynb) for data preprocessing.


## Training
Start training GraphDINO from scratch on ABA dataset:
```
python3 ssl_neuron/main.py --config=ssl_neuron/configs/config.json
```

## Demos
For examples on how to load the data, train the model and perform inference with a pretrained model, see Jupyter notebooks in the [demos folder](https://github.com/marissaweis/ssl_neuron/tree/main/ssl_neuron/demos).


## Citation

If you use this repository in your research, please cite:
```
@article{Weis2021,
      title={Self-supervised Representation Learning of Neuronal Morphologies}, 
      author={Marissa A. Weis and Laura Hansel and Timo Lüddecke and Alexander S. Ecker},
      year={2021},
      journal={arXiv}
}
```
