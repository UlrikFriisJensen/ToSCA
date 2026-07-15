# ToSCA
## Total Scattering Conditioned Variational Autoencoder for Atomic Structure Prediction from Pair Distribution Function Data

#### Ulrik Friis-Jensen $^{1,2}$, Frederik L. Johansen $^{1,2}$, Erik B. Dam $^{2}$, Raghavendra Selvan $^{2}$, and Kirsten M. Ø. Jensen $^{1}$

1: Department of Chemistry & Nano-Science Center, University of Copenhagen <br>
2: Department of Computer Science, University of Copenhagen <br>

### Abstract
Designing new materials is often slow and experimentally intensive, with structural characterization representing a particular bottleneck. Recent advances in machine learning (ML) aim to accelerate this process by predicting plausible atomic structures describing a material directly from experimental data.
Here we present the total scattering conditioned variational autoencoder (ToSCA), a generative ML model for predicting plausible atomic structures of metal oxide nanomaterials from total scattering and pair distribution function (PDF) data. ToSCA learns a chemically meaningful latent representation and predicts unit cells within tolerances suitable for subsequent real-space refinement. The model also shows behavior consistent with learning structural relationships between related crystal types, and can correctly place the new CHILI-Interpolation dataset in the latent space.
Finally, we demonstrate that ToSCA is able to retain useful performance in a realistic use-case by testing it on 11 experimental PDFs from 3 crystal types, including test cases outside of the training distribution.

# Data access

## Access via `CHILI` dataset class

The `CHILI-3K` and `CHILI-Interpolation` datasets can be accessed using the provided dataset class, which can be found [here](https://github.com/UlrikFriisJensen/ToSCA/blob/main/modules/CHILI.py). <br>Ensure that you have the following packages installed before attempting to use the dataset class:
1. **PyTorch**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(https://pytorch.org/get-started/locally/)
2. **PyTorch Geometric**&nbsp;&nbsp;(https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
3. **h5py**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(https://docs.h5py.org/en/stable/build.html)
4. **pandas**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(https://pandas.pydata.org/docs/getting_started/install.html)

Initialising the dataset class, like
```
dataset_object = CHILI(root = 'dataset', dataset = 'CHILI-Interpolation')
```
will automatically download and correctly unpack the `CHILI-Interpolation` dataset into `.pt` files containing PyTorch Geometric Data objects. The raw data is stored in `.h5` files, with each file containing all 5 sizes of nanoparticles generated from the same crystalline material. All data will be stored in the specified `root` directory, here `dataset`.

We have provided an example in this [Python file](https://github.com/UlrikFriisJensen/CHILI/blob/main/distanceregression_example.py) as well as this [Jupyter Notebook](https://github.com/UlrikFriisJensen/CHILI/blob/main/distanceregression_example_notebook.ipynb) with more details on how to use the `CHILI` dataset class to train a basic GCN model on distance regression.

## Direct download
Click on the following hyperlinks to directly download the raw data for [`CHILI-3K`](https://doi.org/10.17894/ucph.e37b6615-8635-49cf-819d-eae60e781a96) and [`CHILI-Interpolation`](https://sid.erda.dk/share_redirect/A8kgn0rf3v).

# Cite
Kindly cite our publication if you use the dataset or any part of the code:
```

```
