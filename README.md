# Tensorflow
Repository for Tensorflow and ANN Tutorials

To run Tensorflow you must have a Python environment set up on your machine.


## Installtion guides:

### Windows:

A python environment is needed to run Tensorflow. There a various ways of setting up a python environment, but you can run into dependency issues with a number of them.

The most straightforward way I have found is to install [Anaconda](https://www.anaconda.com/download/).

Then install tensorflow by running the following commands:
'''
conda create --name tensorflow python=3.5 
activate tensorflow
conda install -c conda-forge tensorflow 
pip install numpy
pip install matplotlib
'''


## Tutorials:

### MNIST:
mnist_tutorial.py - a basic tutorial that builds a simple softmax network for classifying the MNIST dataset.
mnist_tutorial_visualisation.py - visualises the learning of the network.
