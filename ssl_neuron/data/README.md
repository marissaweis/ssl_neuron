# Data Preprocssing

To download the ABA dataset, use the [Allen Software Development Kit](http://alleninstitute.github.io/AllenSDK/cell_types.html). See [demo notebook](http://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html#Cell-Morphology-Reconstructions) on how to use the Allen Cell Types Database.

To get the rotation angles, download Dataset 3 from the Supplementary material of [Gouwens et al. (2019)](https://www.nature.com/articles/s41593-019-0417-0#Sec27). The column "upright_angle" contains the information to rotate the cell to vertical.

See [extract_allen_data.ipynb](https://github.com/marissaweis/ssl_neuron/blob/main/ssl_neuron/data/extract_allen_data.ipynb) for the preprocessing. To speed up training, one can additionally subsample the graphs offline to a smaller number of nodes, i.e. 1000.
