# Data Preprocessing

To download the ABA dataset, use the [Allen Software Development Kit](http://alleninstitute.github.io/AllenSDK/cell_types.html). See [demo notebook](http://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html#Cell-Morphology-Reconstructions) on how to use the Allen Cell Types Database.

To get the rotation angles, download Dataset 3 from the Supplementary material of [Gouwens et al. (2019)](https://www.nature.com/articles/s41593-019-0417-0#Sec27). The column "upright_angle" contains the information to rotate the cell to vertical.

See [extract_allen_data.ipynb](https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/data/extract_allen_data.ipynb) for the preprocessing. To speed up training, one can additionally subsample the graphs offline to a smaller number of nodes, i.e. 1000.


## Data Preprocessing for pretrained model
The pretrained model ([checkpoint](https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/ckpts/)) was trained after the removal of the axons and centering each neuron such that the soma coordinate is (0, 0, 0). Only xyz-coordinates were used as node features.


## Custom data
To utilize GraphDINO with your custom data, you need to specify the directory where your dataset is stored in the config file. Within this directory, there should be a subdirectory named "skeletons". Each sample in your dataset should have its own folder within "skeletons", named after the sample ID. Each of these sample folders should contain two files:

1. "features.npy" - This file stores the node features in a numpy array format with dimensions (number of nodes x number of features).
2. "neighbors.pkl" - This is a Python pickle file containing a dictionary that maps each node ID to the IDs of its neighboring nodes.

Additionally, you'll need to provide lists of sample IDs designated for training and validation. These lists should be stored in files named "train_ids.npy" and "val_ids.npy", respectively, also located within the dataset directory.
