## Setup
```
chmod +x init.sh
./init.sh
```
## MNIST Image Completion
```
THEANO_FLAGS='floatX=float32, device=cuda' python main.py
```

There is another script VAE.py which implements a C-VAE to do the same task.

Main script provides has a few configuration routines. First choice is between a continuous or discrete latent variable. The second choice is between SF and PD estimators. Set device to either cpu or cuda for training.

**Gaussian + PD**: Uses the standard backpropagation routine allowed by the reparametrization trick.

**Gaussian + SF**: Uses the reinforce estimator. Due to its high variance, the network could not converge.

**Discrete + SF**: Again, uses the reinforce estimator for bernoulli latent variable.

**Discrete + PD**: Uses the gumbel-softmax approximation for discrete latent variables. This also allows the usage of reparametrization trick to make use of PD estimators. There are two modes: Soft-sampling and Hard-sampling. Hard-Sampling makes use of the Straight through estimator to propagate gradients through the non-differentiable operation of hard sampling.

## MNIST Classification using Synthetic Gradients (DNI)
```
# train
THEANO_FLAGS='floatX=float32, device=cuda' python dni_classification.py

# test
THEANO_FLAGS='floatX=float32, device=cuda' python dni_classification.py 1 /path/to/stored/weights/
```

For the MNIST Classification, a simple 3-layer NN neural network. Two routines are possible: Using the standard backpropagation and using synthetic gradients.