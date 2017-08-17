## About
The repository hosts work related to the design of a novel estimator for discrete latent variables, leveraging ideas from [synthetic gradients](https://arxiv.org/abs/1608.05343). This repository is a part of work done at [Montreal Institute of Algorithms (MILA)](https://mila.umontreal.ca/) under the mentorship of [Dr. Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/).

## Setup
```
chmod +x init.sh
./init.sh
```
## "Expected" REINFORCE using Synthetic gradients for MNIST Half and Half problem
The problem setup for all scripts remains the same. Given top half of the image, generate the bottom half of the image with a stochastic bottleneck in the middle. For discrete latent variables, there is no counterpart for reparamentrization (as is possible for gaussian continuous latent variables) and hence, getting good gradient estimators is an open problem of sorts. This section deals with the attempts to solve that problem, which leverages idea of synthetic gradients to produce the "expected" value of REINFORCE estimator (which theoretically represents the true gradient). The other scripts in this works are for comparison and baselines using different set of estimators (which are elucidated below). The real test for this method comes from comparison with REINFORCE signal itself, as all other estimators are biased/unscalable.

```
THEANO_FLAGS='floatX=float32, device=cuda' python stochasticdni.py --base_code test # a host of other options with explanations available within script
```

The results are very intriguing. In one aspect, it does show potential to train faster than those achieved with standard REINFORCE (with careful hyperparameter tuning, validation results are significantly faster achieved as well). However, a serious limitation arises from the modelling power of the subnetworks, which usually forces a premature saturation of the model being trained. This direction of research is promising (albeit exhausting because the interplay of the subnetwork and main network is not clearly understood, and hence fairly heuristical). However, more efforts are required before this method can be considered as a serious replacement for REINFORCE. Note that the standard benefits of decoupling associated with synthetic gradients can be applicable over here as well.

## Some gradient estimators for MNIST Half and Half problem
```
THEANO_FLAGS='floatX=float32, device=cuda' python main.py --base_code test # a host of other options with explanations available within script
```

Main script provides a lot of training options which are explained in the script (or can be accessed using help of argparse). Two major choices are: Latent variable (Either discrete or continuous) or Gradient estimator (Score Function estimator or SF, popularly known as REINFORCE and reparametrized backpropagation/path derivative estimators).

  - **Gaussian latent variable with backpropagation**: Uses the standard backpropagation routine allowed by the reparametrization trick.

  - **Gaussian latent variable with REINFORCE**: Uses the REINFORCE estimator. Due to its high variance, the network will struggle to converge.

  - **Discrete latent variable with REINFORCE**: Uses the REINFORCE estimator for bernoulli latent variable.

  - **Discrete latenet variable with ST**: Uses the straight through estimator for bernoulli latent variable.

  - **Discrete latent variable with continuous relaxation using Gumbel-Softmax reparametrization**: Uses the gumbel-softmax approximation for discrete latent variables. This also allows the usage of reparametrization trick to make use of PD estimators. There are two modes: Soft-sampling and Hard-sampling. Hard-Sampling makes use of the Straight Through (ST) estimator to propagate gradients through the non-differentiable operation of hard sampling.

Note: REINFORCE estimators have a conditional mean baseline to reduce variance (by default), which can be changed as well. 
## MNIST Classification using Synthetic Gradients (DNI)
```
# train
THEANO_FLAGS='floatX=float32, device=cuda' python dni_classification.py

# test
THEANO_FLAGS='floatX=float32, device=cuda' python dni_classification.py 1 /path/to/stored/weights/
```

For the MNIST Classification, a simple 3-layer NN neural network is trained. Two routines are possible: Using the standard backpropagation or using synthetic gradients. The model trained using this script achieves 2.4% error rate (which comes very close to the error rate 2.2% reported in the original paper). Note: the network in this script is trained without batch-normalization (for no good reason).