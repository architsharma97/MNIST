#!/bin/bash

mkdir MNIST/
cd MNIST/
wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
wget "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
wget "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

cd ../
mkdir Results_VAE/
mkdir Results_VAE/PD
mkdir Results_VAE/SF

mkdir Results/
mkdir Results/cont
mkdir Results/disc
# estimators for continuous latent variables
mkdir Results/cont/PD/
mkdir Results/cont/SF/

# estimators for discrete latent variables
mkdir Results/disc/SF/
mkdir Results/disc/PD/
mkdir Results/disc/synthetic_gradients/
mkdir Results/disc/ST/

# results for classification
mkdir Results/classification/
mkdir Results/classification/backprop
mkdir Results/classification/synthetic_gradients