import sys

import numpy as np
from read_mnist import read, show
import theano
import theano.tensor as T
from utils import init_weights, _concat
import argparse

from adam import adam
from sgd import SGD

from collections import OrderedDict
import time

'''
Experiments with synthetic gradients in stochastic graphs. The task is to complete images from MNIST, when only the top half is given.
Bernoulli latent variables with REINFORCE estimators are the baseline for comparison.
'''
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='train', help='train or test')
parser.add_argument('-g', '--target', type=str, default='REINFORCE', help='Either ST or REINFORCE')

# meta-parameters, governs networks and their training
parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of samples per training example for SF estimator')
parser.add_argument('-u', '--update_style', type=str, default='fixed', 
					help='Either (decay) or (fixed). Decay will increase the number of iterations after which the subnetwork is updated.')
parser.add_argument('-x', '--sg_type',type=str, default='lin_deep', 
					help='Type of synthetic gradient subnetwork: linear (lin) or a two-layer nn (deep) or both (lin_deep)')
parser.add_argument('-y', '--sg_inp', type=str, default='11111',
					help='Customize input to synthetic subnetworks: Construct a string of 0,1 with 1 at inputs to be conditioned on')
parser.add_argument('-z', '--bn_type', type=int, default=1,
					help='0: BN->Matrix Multiplication->Nonlinearity, 1: Matrix Multiplication->BN->Nonlinearity')
parser.add_argument('-w', '--max_grad', type=float, default=0.0,
					help='Maximum elementwise mean squared norm of the target signal, otherwise gradient is clipped')
# update frequencies
parser.add_argument('-q', '--main_update_freq', type=int, default=1,
					help='Number of iterations after which the main network is updated')
parser.add_argument('-i', '--sub_update_freq', type=int, default=1,
					help='Number of iterations after which the deep subnetwork is updated')

# REINFORCE only meta-parameters
parser.add_argument('-v', '--var_red', type=str, default='cmr',
					help='Use different control variates for targets, unconditional mean (mr) and conditional mean (cmr)')
parser.add_argument('-aa', '--use_exp_reward', type=int, default=0,
					help='Use a exponential reward, which allows for smaller gradients to be learnt (1 if yes, 0 otherwise)')
parser.add_argument('-d', '--exptemp', type=float, default=100.0,
					help='Rewards are exponentiated: This controls the temperature.')

# while testing
parser.add_argument('-l', '--load', type=str, default=None, help='Path to weights')

# hyperparameters
parser.add_argument('-a', '--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=100, help='Size of the minibatch used for training')
parser.add_argument('-j', '--dropout_prob', type=float, default=0.5, help='Probability with which neuron is dropped')
parser.add_argument('-k', '--sg_reg', type=float, default=0.0, help='L2 regularization of subnetwork cost')

# additional training and saving related arguments
parser.add_argument('-t', '--term_condition', type=str, default='epochs', 
					help='Training terminates either when number of epochs are completed (epochs) or when minimum cost is achieved for a batch (mincost)')
parser.add_argument('-n', '--num_epochs', type=int, default=100, 
					help='Number of epochs, to be specified when termination condition is epochs')
parser.add_argument('-c', '--min_cost', type=float, default=55.0, 
					help='Minimum cost to be achieved for a minibatch, to be specified when termination condition is mincost')
parser.add_argument('-s', '--save_freq', type=int, default=100, 
					help='Number of epochs after which weights should be saved')
parser.add_argument('-f', '--base_code', type=str, default='sg',
					help='A unique identifier for saving purposes')

# miscellaneous
parser.add_argument('-e', '--random_seed', type=int, default=42, help='Seed to initialize random streams')
parser.add_argument('-o', '--latent_type', type=str, default='disc', help='No other options')
parser.add_argument('-p', '--clip_probs', type=int, default=0,
					help='clip latent probabilities (1) or not (0), useful for testing training under NaNs')

args = parser.parse_args()

# initialize random streams
if "gpu" in theano.config.device:
	srng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=args.random_seed)
else:
	srng = T.shared_randomstreams.RandomStreams(seed=args.random_seed)


code_name = args.base_code + '_' + str(args.repeat) + '_gc' + str(args.max_grad)

estimator = 'synthetic_gradients'

delta = 1e-7
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# converts images into binary images for simplicity
def binarize_img(img):
	return np.asarray(img >= 100, dtype=np.int8)

# split into two images
def split_img(img):
	# images are flattened into a vector: just need to split into half
	veclen = len(img)
	return (img[:veclen/2], img[veclen/2:])

def param_init_fflayer(params, prefix, nin, nout, zero_init=False, batchnorm=False):
	'''
	Initializes weights for a feedforward layer
	'''
	global args
	if zero_init:
		params[_concat(prefix, 'W')] = np.zeros((nin, nout)).astype('float32')
	else:
		params[_concat(prefix, 'W')] = init_weights(nin, nout, type_init='ortho')
	
	params[_concat(prefix, 'b')] = np.zeros((nout,)).astype('float32')
	
	if batchnorm:
		if args.bn_type == 0:
			dim = nin
		else:
			dim = nout
		params[_concat(prefix, 'g')] = np.ones((dim,), dtype=np.float32)
		params[_concat(prefix, 'be')] = np.zeros((dim,)).astype('float32')
		params[_concat(prefix, 'rm')] = np.zeros((1, dim)).astype('float32')
		params[_concat(prefix, 'rv')] = np.ones((1, dim), dtype=np.float32)
	
	return params

def fflayer(tparams, state_below, prefix, nonlin='tanh', batchnorm=None, dropout=None):
	'''
	A feedforward layer
	Note: None means dropout/batch normalization is not used.
	Use 'train' or 'test' options.
	'''
	global srng, args

	# apply batchnormalization on the input
	if args.bn_type == 0:
		inp = state_below
	else:
		inp = T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')]

	if batchnorm == 'train':
		axes = (0,)
		mean = inp.mean(axes, keepdims=True)
		var = inp.var(axes, keepdims=True)
		invstd = T.inv(T.sqrt(var + 1e-4))
		inp = (inp - mean) * tparams[_concat(prefix, 'g')] * invstd + tparams[_concat(prefix, 'be')]
		
		running_average_factor = 0.1	
		m = T.cast(T.prod(inp.shape) / T.prod(mean.shape), 'float32')
		tparams[_concat(prefix, 'rm')] = tparams[_concat(prefix, 'rm')] * (1 - running_average_factor) + mean * running_average_factor
		tparams[_concat(prefix, 'rv')] = tparams[_concat(prefix, 'rv')] * (1 - running_average_factor) + (m / (m - 1)) * var * running_average_factor
		
	elif batchnorm == 'test':
		inp = (inp - tparams[_concat(prefix, 'rm')].flatten()) * tparams[_concat(prefix, 'g')] / T.sqrt(tparams[_concat(prefix, 'rv')].flatten() + 1e-4) + tparams[_concat(prefix, 'be')]
	
	if args.bn_type == 0:
		preact = T.dot(inp, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')]
	else:
		preact = inp

	# dropout is carried out with fixed probability
	if dropout == 'train':
		dropmask = srng.binomial(n=1, p=1.-args.dropout_prob, size=preact.shape, dtype=theano.config.floatX)
		preact *= dropmask
	
	elif dropout == 'test':
		preact *= 1. - args.dropout_prob

	if nonlin == None:
		return preact
	elif nonlin == 'tanh':
		return T.tanh(preact)
	elif nonlin == 'sigmoid':
		return T.nnet.nnet.sigmoid(preact)
	elif nonlin == 'softplus':
		return T.nnet.nnet.softplus(preact)
	elif nonlin == 'relu':
		return T.nnet.nnet.relu(preact)

def param_init_sgmod(params, prefix, units, zero_init=True):
	'''
	Initialization for synthetic gradient subnetwork
	'''
	global args
	
	# conditioned on the whole image, on the activation produced by encoder input and the backpropagated gradients for latent samples.
	inp_list = [14*28, 14*28, units, units, units]
	inp_size = 0
	for i in range(5):
		if args.sg_inp[i] == '1':
			inp_size += inp_list[i]

	if not zero_init:
		if args.sg_type == 'lin':
			params[_concat(prefix, 'W')] = init_weights(inp_size, units, type_init='ortho')
			params[_concat(prefix, 'b')] = np.zeros((units,)).astype('float32')

	else:
		if args.sg_type == 'lin' or args.sg_type == 'lin_deep':
			params[_concat(prefix, 'W')] = np.zeros((inp_size, units)).astype('float32')
			params[_concat(prefix, 'b')] = np.zeros((units,)).astype('float32')

		if args.sg_type == 'deep' or args.sg_type == 'lin_deep':
			params = param_init_fflayer(params, _concat(prefix, 'I'), inp_size, 1024, batchnorm=True)
			params = param_init_fflayer(params, _concat(prefix, 'H'), 1024, 1024, batchnorm=True)
			if args.bn_type == 0:
				params = param_init_fflayer(params, _concat(prefix, 'o'), 1024, units, zero_init=True, batchnorm=True)
			else:
				params = param_init_fflayer(params, _concat(prefix, 'o'), 1024, units, zero_init=True, batchnorm=False)

		if args.sg_type == 'custom':
			# residual block
			params[_concat(prefix, 'W')] = np.zeros((inp_size, units)).astype('float32')
			params[_concat(prefix, 'b')] = np.zeros((units,)).astype('float32')
			params[_concat(prefix, 'g')] = np.ones((units,), dtype=np.float32)
			params[_concat(prefix, 'be')] = np.zeros((units,)).astype('float32')

			params = param_init_fflayer(params, _concat(prefix, '1'), units, 1024, batchnorm=True)
			params = param_init_fflayer(params, _concat(prefix, '2'), 1024, units, zero_init=True)

	return params

def synth_grad(tparams, prefix, inp, mode='Train'):
	'''
	Synthetic gradients
	'''
	global args
	# depending on the bn type being used, bn is used/not used in the layer
	if args.bn_type == 0:
		bn_last = 'train'
	else:
		bn_last = None

	if args.sg_type == 'lin':
		return T.dot(inp, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')]
	
	elif args.sg_type == 'deep' or args.sg_type == 'lin_deep':
		outi = fflayer(tparams, inp, _concat(prefix, 'I'), nonlin='relu', batchnorm='train', dropout=None)
		outh = fflayer(tparams, outi, _concat(prefix,'H'), nonlin='relu', batchnorm='train', dropout=None)
		
		if args.sg_type == 'deep':
			return fflayer(tparams, outh + outi, _concat(prefix, 'o'), batchnorm=bn_last, nonlin=None)
		elif args.sg_type == 'lin_deep':
			return T.dot(inp, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')] + fflayer(tparams, outh + outi, _concat(prefix, 'o'), batchnorm=bn_last, nonlin=None)
	
	elif args.sg_type == 'custom':
		# channel 2 which forms the skip connection
		inp = T.dot(inp, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')]
		
		# channel 1
		mean = inp.mean((0,), keepdims=True)
		var = inp.var((0,), keepdims=True)
		invstd = T.inv(T.sqrt(var + 1e-4))
		out1 = T.nnet.nnet.relu((inp - mean) * tparams[_concat(prefix, 'g')] * invstd + tparams[_concat(prefix, 'be')])
		out2 = fflayer(tparams, out1, _concat(prefix, '1'), nonlin='relu', batchnorm='train')

		# channel 1 + channel 2
		return fflayer(tparams, out2, _concat(prefix, '2'), nonlin=None) + inp
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

print "Creating partial images"
# collect training data and converts image into binary and does row major flattening
trc = np.asarray([binarize_img(img).flatten() for lbl, img in read(dataset='training', path ='MNIST/')], dtype=np.float32)

# collect test data and converts image into binary and does row major flattening
tec = np.asarray([binarize_img(img).flatten() for lbl, img in read(dataset='testing', path = 'MNIST/')], dtype=np.float32)

# split images
trp = [split_img(img) for img in trc]
tep = [split_img(img) for img in tec]

print "Initializing parameters"
# parameter initializations
ff_e = 'ff_enc'
ff_d = 'ff_dec'
sg = 'sg'
latent_dim = 50

params = OrderedDict()

# encoder
params = param_init_fflayer(params, _concat(ff_e, 'i'), 14*28, 200, batchnorm=True)
params = param_init_fflayer(params, _concat(ff_e, 'h'), 200, 100, batchnorm=True)

# latent
if args.bn_type == 0:
	params = param_init_fflayer(params, _concat(ff_e, 'bern'), 100, latent_dim, batchnorm=True)
else:
	params = param_init_fflayer(params, _concat(ff_e, 'bern'), 100, latent_dim, batchnorm=False)

# synthetic gradient module for the last encoder layer
params = param_init_sgmod(params, _concat(sg, 'r'), latent_dim)

# loss prediction neural network, conditioned on input and output (in this case the whole image), acts as the baseline
if args.target == 'REINFORCE' and args.var_red == 'cmr':	
	params = param_init_fflayer(params, 'loss_pred', 28*28, 1)

# decoder parameters
params = param_init_fflayer(params, _concat(ff_d, 'n'), latent_dim, 100)
params = param_init_fflayer(params, _concat(ff_d, 'h'), 100, 200, batchnorm=True)

if args.bn_type == 0 :
	params = param_init_fflayer(params, _concat(ff_d, 'o'), 200, 14*28, batchnorm=True)
else:
	params = param_init_fflayer(params, _concat(ff_d, 'o'), 200, 14*28, batchnorm=False)

# restore from saved weights
if args.load is not None:
	lparams = np.load(args.load)

	for key, val in lparams.iteritems():
		params[key] = val

tparams = OrderedDict()
for key, val in params.iteritems():
	tparams[key] = theano.shared(val, name=key)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# Training graph
if args.mode == 'train':
	print "Constructing graph for training"
	# create shared variables for dataset for easier access
	top = np.asarray([splitimg[0] for splitimg in trp], dtype=np.float32)
	bot = np.asarray([splitimg[1] for splitimg in trp], dtype=np.float32)
	
	train = theano.shared(top, name='train')
	train_gt = theano.shared(bot, name='train_gt')

	# pass a batch of indices while training
	img_ids = T.vector('ids', dtype='int64')
	img = train[img_ids,:]
	img_r = T.extra_ops.repeat(img, args.repeat, axis=0)

	# repeat args.repeat-times to compensate for the sampling process
	gt_unrepeated = train_gt[img_ids,:]
	gt = T.extra_ops.repeat(gt_unrepeated, args.repeat, axis=0)

	# inputs for synthetic gradient networks: needs the top half of the image batch
	target_gradients = T.matrix('tg', dtype='float32')
	activation = T.matrix('sg_input_probs', dtype='float32')
	latent_gradients = T.matrix('sg_input_latgrads', dtype='float32')
	samples = T.matrix('sg_input_samples', dtype='float32')

# Test graph
else:
	print "Constructing the test graph"
	# create shared variables for dataset for easier access
	top = np.asarray([splitimg[0] for splitimg in tep], dtype=np.float32)
	bot = np.asarray([splitimg[1] for splitimg in tep], dtype=np.float32)
	
	test = theano.shared(top, name='train')
	test_gt = theano.shared(bot, name='train_gt')

	# image ids
	img_ids = T.vector('ids', dtype='int64')
	img = test[img_ids,:]
	gt = test_gt[img_ids,:]

# encoding
out1 = fflayer(tparams, img, _concat(ff_e, 'i'), batchnorm=args.mode)
out2 = fflayer(tparams, out1, _concat(ff_e, 'h'), batchnorm=args.mode)
if args.bn_type == 0:
	out3 = fflayer(tparams, out2, _concat(ff_e, 'bern'), nonlin='sigmoid', batchnorm=args.mode)
else:
	out3 = fflayer(tparams, out2, _concat(ff_e, 'bern'), nonlin='sigmoid', batchnorm=None)

if args.clip_probs == 0:
	latent_probs = out3
else:
	latent_probs = T.clip(out3, delta, 1. - delta)

if args.mode == 'test' or args.target == 'REINFORCE':
	# sample a bernoulli distribution, which a binomial of 1 iteration
	latent_samples = srng.binomial(size=latent_probs.shape, n=1, p=latent_probs, dtype=theano.config.floatX)

elif args.target == 'ST':
	# sample a bernoulli distribution, which a binomial of 1 iteration
	latent_samples_uncorrected = srng.binomial(size=latent_probs.shape, n=1, p=latent_probs, dtype=theano.config.floatX)
	
	# for stop gradients trick
	dummy = latent_samples_uncorrected - latent_probs
	latent_samples = latent_probs + dummy

# decoding
outz = fflayer(tparams, latent_samples, _concat(ff_d, 'n'))
outh = fflayer(tparams, outz, _concat(ff_d, 'h'), batchnorm=args.mode)
if args.bn_type == 0:
	probs = fflayer(tparams, outh, _concat(ff_d, 'o'), nonlin='sigmoid', batchnorm=args.mode)
else:
	probs = fflayer(tparams, outh, _concat(ff_d, 'o'), nonlin='sigmoid', batchnorm=None)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# Training
if args.mode == 'train':
	# --------------------Gradients for main network----------------------------------------------------------------------------
	reconstruction_loss = T.nnet.binary_crossentropy(probs, gt).mean(axis=1)
	print "Computing synthetic gradients"

	# separate parameters for encoder, decoder and sg subnetworks
	param_dec = [val for key, val in tparams.iteritems() if ('ff_dec' in key) and ('rm' not in key and 'rv' not in key)]
	param_enc = [val for key, val in tparams.iteritems() if ('ff_enc' in key) and ('rm' not in key and 'rv' not in key)]
	param_sg = [val for key, val in tparams.iteritems() if ('sg' in key) and ('rm' not in key and 'rv' not in key)]

	print "Encoder parameters:", param_enc
	print "Decoder parameters:", param_dec
	print "Synthetic gradient subnetwork parameters:", param_sg
	
	print "Computing gradients wrt to decoder parameters"
	cost_decoder = T.mean(reconstruction_loss)
	
	# regularization
	weights_sum_dec = 0.
	for val in param_dec:
		weights_sum_dec += (val**2).sum()
	grads_decoder = T.grad(cost_decoder, wrt=param_dec)
	
	# for better estimation, converts into a learnt straight through estimator with ST/REINFORCE
	gradz = T.grad(cost_decoder, wrt=latent_samples)
	
	print "Computing gradients wrt to encoder parameters"
	known_grads = OrderedDict()
	var_list = [img_r, gt, latent_probs, gradz, latent_samples]
	sg_cond_vars_actual = [var_list[i] for i in range(5) if args.sg_inp[i] == '1']
	known_grads[out3] = synth_grad(tparams, _concat(sg, 'r'), T.concatenate(sg_cond_vars_actual, axis=1), mode='test').reshape((args.batch_size, args.repeat, latent_dim)).sum(axis=1) / args.repeat
	grads_encoder = T.grad(None, wrt=param_enc, known_grads=known_grads)

	# combine in this order only
	grads_net = grads_encoder + grads_decoder
	
	# ---------------Gradients for synthetic gradient network-------------------------------------------------------------------
	if args.target == 'REINFORCE':
		print "Getting REINFORCE target"
		# arguments to be considered constant when computing gradients
		consider_constant = [reconstruction_loss, latent_samples]

		if args.var_red is None:
			cost_encoder = T.mean(reconstruction_loss * T.switch(latent_samples, T.log(latent_probs), T.log(1. - latent_probs)).sum(axis=1))

		elif args.var_red == 'mr':
			# unconditional mean is subtracted from the reconstruction loss, to yield a relatively lower variance unbiased REINFORCE estimator
			cost_encoder = T.mean((reconstruction_loss - T.mean(reconstruction_loss)) * T.switch(latent_samples, T.log(latent_probs), T.log(1. - latent_probs)).sum(axis=1))
			
		elif args.var_red == 'cmr':
			# conditional mean is subtracted from the reconstruction loss to lower variance further
			baseline = T.extra_ops.repeat(fflayer(tparams, T.concatenate([img, gt_unrepeated], axis=1), 'loss_pred', nonlin='relu'), args.repeat, axis=0)
			
			if args.use_exp_reward:
				cost_encoder = T.mean(-(T.exp(-reconstruction_loss / args.exptemp) - baseline.T) * T.switch(latent_samples, T.log(latent_probs), T.log(1. - latent_probs)).sum(axis=1))
				cost_pred = T.mean((T.exp(-reconstruction_loss / args.exptemp) - baseline.T) ** 2)
			else:
				cost_encoder = T.mean((reconstruction_loss - baseline.T) * T.switch(latent_samples, T.log(latent_probs), T.log(1. - latent_probs)).sum(axis=1))
				cost_pred = T.mean((reconstruction_loss - baseline.T) ** 2)

			params_loss_predictor = [val for key, val in tparams.iteritems() if 'loss_pred' in key]
			print "Loss predictor parameters:", params_loss_predictor

			grads_plp = T.grad(cost_pred, wrt=params_loss_predictor, consider_constant=[reconstruction_loss])
			consider_constant += [baseline]
			grads_net = grads_encoder + grads_plp + grads_decoder

		# computing target for synthetic gradient, will be output in every iteration
		sg_target = T.grad(cost_encoder, wrt=out3, consider_constant=consider_constant)

	elif args.target == 'ST':
		print "Getting ST target"
		sg_target = T.grad(cost_decoder, wrt=out3, consider_constant=[dummy])
	
	# regularization
	weights_sum_sg = 0.
	for val in param_sg:
		weights_sum_sg += (val**2).sum()

	# normalize target_gradients to have an upper bound on the norm
	if args.max_grad > 0.0:
		target_gradients_clip = T.switch(T.mean(target_gradients ** 2) < args.max_grad, target_gradients, target_gradients * T.sqrt(args.max_grad / T.mean(target_gradients ** 2)))
	else:
		print "No gradient clipping"
		target_gradients_clip = target_gradients
	
	var_list = [img_r, gt, activation, latent_gradients, samples]
	sg_cond_vars_symbol = [var_list[i] for i in range(5) if args.sg_inp[i] == '1']
	loss_sg = T.mean((target_gradients_clip - synth_grad(tparams, _concat(sg, 'r'), T.concatenate(sg_cond_vars_symbol, axis=1)).reshape((args.batch_size, args.repeat, latent_dim)).sum(axis=1) / args.repeat) ** 2)
	grads_sg = T.grad(loss_sg + args.sg_reg * weights_sum_sg, wrt=param_sg)
	tgnorm = T.mean(target_gradients_clip ** 2)
	# ----------------------------------------------General training routine------------------------------------------------------
	
	lr = T.scalar('lr', dtype='float32')
	cost = cost_decoder

	inps_net = [img_ids]
	inps_sg = inps_net + [target_gradients] + var_list[2:]
	tparams_net = OrderedDict()
	tparams_sg = OrderedDict()
	for key, val in tparams.iteritems():
		print key
		# running means and running variances should not be included in the list for which gradients are computed
		if 'rm' in key or 'rv' in key:
			continue
		elif 'sg' in key:
			tparams_sg[key] = val
		else:
			tparams_net[key] = val

	print "Setting up optimizers"
	f_grad_shared, f_update = adam(lr, tparams_net, grads_net, inps_net, [cost, sg_target, latent_probs, gradz, latent_samples])
	# f_grad_shared_sg, f_update_sg = adam(lr, tparams_sg, grads_sg, inps_sg, [loss_sg, tgnorm])

	# sgd with momentum updates
	sgd = SGD(lr=args.learning_rate)
	f_update_sg = theano.function(inps_sg, [loss_sg, tgnorm], updates=sgd.get_grad_updates(loss_sg, param_sg), on_unused_input='ignore', profile=False)
	
	print "Training"
	cost_report = open('./Results/' + args.latent_type + '/' + estimator + '/tsgd_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '.txt', 'w')
	id_order = range(len(trc))

	iters = 0
	min_cost = 100000.0
	epoch = 0
	condition = False
	
	# reinitialization scheme for channel 1 of custom subnetwork
	# gam = T.vector('rg', dtype='float32')
	# bet = T.vector('rb', dtype='float32')
	# w1 = T.matrix('rw1', dtype='float32')
	# w2 = T.matrix('rw2', dtype='float32')
	# b1 = T.vector('rb1', dtype='float32')
	# b2 = T.vector('rb2', dtype='float32')

	# reinit_dict = [(tparams['sg_r_g'], gam), (tparams['sg_r_be'], bet), (tparams['sg_r_1_W'], w1), (tparams['sg_r_2_W'], w2), (tparams['sg_r_1_b'], b1), (tparams['sg_r_2_b'], b2)]
	# reinit = theano.function([gam, bet, w1, w2, b1, b2], None, updates=reinit_dict)

	while condition == False:
		print "Epoch " + str(epoch + 1),

		np.random.shuffle(id_order)
		epoch_cost = 0.
		epoch_cost_sg = 0.
		epoch_start = time.time()
		for batch_id in range(len(trc)/args.batch_size):
			batch_start = time.time()
			iters += 1

			idlist = id_order[batch_id*args.batch_size:(batch_id+1)*args.batch_size]
			
			# main network update
			outs = f_grad_shared(idlist)
			cost, t = outs[:2]
			if iters % args.main_update_freq == 0:
				f_update(args.learning_rate)
			
			# subnetwork update
			cost_sg = 'NC'
			tmag = 'NC'
			if iters % args.sub_update_freq == 0 and not np.isnan((t**2).mean()):
				cost_sg, tmag = f_update_sg(idlist, *outs[1:])
				# f_update_sg(args.learning_rate)

				epoch_cost_sg += cost_sg
			
			elif np.isnan((t**2).mean()):
				print "NaN encountered at", iters

			# decay mode
			if args.update_style == 'decay':
				 if iters == 2000:
				 	args.sub_update_freq = 2
				 elif iters == 5000:
				 	args.sub_update_freq = 5
				 elif iters == 10000:
				 	args.sub_update_freq = 10
				 elif iters == 20000:
				 	args.sub_update_freq = 50
				 elif iters == 30000:
			 		args.sub_update_freq = 100
			
			epoch_cost += cost
			min_cost = min(min_cost, cost)
			cost_report.write(str(epoch) + ',' + str(batch_id) + ',' + str(cost) + ',' + str(cost_sg) + ',' + str(tmag) + ',' + str(time.time() - batch_start) + '\n')

			# partial REINITIALIZATION of the subnetwork to get it out of the local minima
			# if iters == 1:
			# 	smooth_sg_cost = cost_sg
			# 	flag = False
			# else:
			# 	smooth_sg_cost = 0.99 * smooth_sg_cost + 0.01 * cost_sg

			# if (smooth_sg_cost >= 0.4 and flag == True):
			# 	print "Reinitializing the channel 1 for subnetwork"
			# 	params = param_init_fflayer(params, _concat('sg_r', '1'), latent_dim, 1024, batchnorm=True)
			# 	params = param_init_fflayer(params, _concat('sg_r', '2'), 1024, latent_dim, zero_init=True)
			# 	reinit(params['sg_r_g'], params['sg_r_be'], params['sg_r_1_W'], params['sg_r_2_W'], params['sg_r_1_b'], params['sg_r_2_b'])
			# 	flag = False
			# elif smooth_sg_cost <=0.4 and flag == False:
			# 	print "Entered in susceptible zone"
			# 	flag = True
		
		print ": Cost " + str(epoch_cost) + " : SG Cost " + str(epoch_cost_sg) + " : Time " + str(time.time() - epoch_start)
		
		# save every args.save_freq epochs
		if (epoch + 1) % args.save_freq == 0:
			print "Saving..."

			params = {}
			for key, val in tparams.iteritems():
				params[key] = val.get_value()

			# numpy saving
			np.savez('./Results/' + args.latent_type + '/' + estimator + '/tsgd_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(epoch+1) + '.npz', **params)
			print "Done!"

		epoch += 1
		if args.term_condition == 'mincost' and min_cost < args.min_cost:
			condition = True
		elif args.term_condition == 'epochs' and epoch >= args.num_epochs:
			condition = True

	# saving the final model
	if epoch % args.save_freq != 0:
		print "Saving..."

		params = {}
		for key, val in tparams.iteritems():
			params[key] = val.get_value()

		# numpy saving
		np.savez('./Results/' + args.latent_type + '/' + estimator + '/tsgd_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(epoch) + '.npz', **params)
		print "Done!"

# Test
else:
	prediction = probs > 0.5

	loss = abs(prediction-gt).sum()

	# compiling test function
	inps = [img_ids]
	f = theano.function(inps, [prediction, loss])
	idx = 10
	pred, loss = f([idx])

	show(tec[idx].reshape(28,28))

	reconstructed_img = np.zeros((28*28,))
	reconstructed_img[:14*28] = tep[idx][0]
	reconstructed_img[14*28:] = pred
	show(reconstructed_img.reshape(28,28))
	print loss