# Tensorflow
Repository for Tensorflow and ANN Tutorials

To run Tensorflow you must have a Python environment set up on your machine.


## Installtion guides

### Windows:

A python environment is needed to run Tensorflow. There a various ways of setting up a python environment, but you can run into dependency issues with a number of them.

The most straightforward way I have found is to install [Anaconda](https://www.anaconda.com/download/).

Then install tensorflow by running the following commands (press y then enter when prompted during the installation):
```
conda create --name tensorflow python=3.5 
activate tensorflow
conda install -c conda-forge tensorflow
python -m pip install --upgrade pip
pip install numpy
pip install matplotlib
```
This installation can be tested by running the command:
```
python -c "import tensorflow as tf; import numpy as np; import matplotlib as plt; print('Its Alive!')"
```

If "Its Alive!" appears in the console the installation was successfull.


A useful Python IDE is Spyder, which can be installed with the following command:
```
conda install -c anaconda spyder
```
And then run simply with:
```
spyder
```




## Tutorials

### MNIST:
An introductory course into tensorflow. Uses the MNIST dataset to train a simple (single layer) sigmoid network on handwritten digit recognition.
