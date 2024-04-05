# Self-supervised Representation Learning for Neuronal Morphologies

This repository contains code to the paper [Self-Supervised Graph Representation Learning for Neuronal Morphologies](https://openreview.net/forum?id=ThhMzfrd6r) by M.A. Weis, L. Hansel, T. Lüddecke and A.S. Ecker (2023).


## System Requirements
### Hardware requirements
The training of GraphDINO requires a GPU. All trainings for the publication were performed on a NVIDIA Quadro RTX 5000 single GPU. Training on the neuronal BBP dataset ran for approximately 10 hours for 100,000 training iterations.

### Software requirements
#### OS Requirements
The code was developed and tested on Linux (Ubuntu 16.04).

#### Python Dependencies
The Python Dependencies are specified in [setup.py](https://github.com/marissaweis/ssl_neuron/blob/main/setup.py).


## Installation
```
python3 setup.py install
```


## Data

Extract data using the [Allen Software Development Kit](http://alleninstitute.github.io/AllenSDK/cell_types.html). See [demo notebook](http://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html#Cell-Morphology-Reconstructions) on how to use the Allen Cell Types Database.

See [extract_allen_data.ipynb](https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/data/extract_allen_data.ipynb) for data preprocessing.

See [Data README](https://github.com/marissaweis/ssl_neuron/tree/main/ssl_neuron/data#readme) for instructions on how to use GraphDINO with your custom dataset.

## Training
Start training GraphDINO from scratch on ABA dataset:
```
python3 ssl_neuron/main.py --config=ssl_neuron/configs/config.json
```

The training code will write checkpoint files of the model weights to the checkpoint directory specified in the config file.


## Demos
For examples on how to load the data, train the model and perform inference with a pretrained model, see Jupyter notebooks in the [demos folder](https://github.com/marissaweis/ssl_neuron/tree/main/ssl_neuron/demos).


## Citation

If you use this repository in your research, please cite:
```
@article{Weis2023,
      title={Self-Supervised Graph Representation Learning for Neuronal Morphologies},
      author={Marissa A. Weis and Laura Hansel and Timo L{\"u}ddecke and Alexander S. Ecker},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2023}
}
```


## License
This project is covered under the MIT License.
