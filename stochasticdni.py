import sys

import numpy as np
from read_mnist import read, show
import theano
import theano.tensor as T
from utils import init_weights, _concat
from adam import adam
import argparse

from collections import OrderedDict
import time

'''
Experiments with synthetic gradients in stochastic graphs. The task is to complete images from MNIST, when only the top half is given.
Bernoulli latent variables with REINFORCE estimators are the baseline for comparison.
'''
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='train', help='train or test')

# meta-parameters, governs networks and their training
parser.add_argument('-r', '--repeat', type=int, default=1, help='Number of samples per training example for SF estimator')
parser.add_argument('-u', '--update_style', type=str, default='fixed', 
					help='Either (decay) or (fixed). Decay will increase the number of iterations after which the subnetwork is updated.')
parser.add_argument('-x', '--sg_type',type=str, default='lin', 
					help='Type of synthetic gradient subnetwork: linear (lin) or a two-layer nn (deep) or both (lin_deep)')
parser.add_argument('-v', '--var_red', type=str, default='cmr',
					help='Use different control variates for targets, unconditional mean (mr) and conditional mean (cmr)')

# while testing
parser.add_argument('-l', '--load', type=str, default=None, help='Path to weights')

# hyperparameters
parser.add_argument('-a', '--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=100, help='Size of the minibatch used for training')

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
parser.add_argument('-p', '--clip_probs', type=int, default=1,
					help='clip latent probabilities (1) or not (0), useful for testing training under NaNs')

args = parser.parse_args()

# initialize random streams
if "gpu" in theano.config.device:
	srng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=args.random_seed)
else:
	srng = T.shared_randomstreams.RandomStreams(seed=args.random_seed)


code_name = args.base_code + '_' + str(args.repeat)

estimator = 'synthetic_gradients'

delta = 1e-10
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
	if zero_init:
		params[_concat(prefix, 'W')] = np.zeros((nin, nout)).astype('float32')
	else:
		params[_concat(prefix, 'W')] = init_weights(nin, nout, type_init='ortho')
	
	params[_concat(prefix, 'b')] = np.zeros((nout,)).astype('float32')
	
	if batchnorm:
		params[_concat(prefix, 'g')] = np.ones((nout,), dtype=np.float32)
		params[_concat(prefix, 'be')] = np.zeros((nout,)).astype('float32')
	
	return params

def fflayer(tparams, state_below, prefix, nonlin='tanh', batchnorm=False):
	'''
	A feedforward layer
	'''
	preact = T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')]
	
	# currently only valid for batchnorm in training
	if batchnorm:
		axes = (0,)
		mean = preact.mean(axes, keepdims=True)
		var = preact.var(axes, keepdims=True)
		invstd = T.inv(T.sqrt(var + 1e-4))
		preact = (preact - mean) * tparams[_concat(prefix, 'g')] * invstd + tparams[_concat(prefix, 'be')]

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
	Initializes a linear regression based model for estimating gradients, conditioned on the class labels
	'''
	global args
	# conditioned on the whole image, on the activation produced by encoder input and the backpropagated gradients for latent samples.
	inp_size = 28*28 + units + units
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
			params = param_init_fflayer(params, _concat(prefix, '1'), 1024, 1024, batchnorm=True)
			params = param_init_fflayer(params, _concat(prefix, '2'), 1024, 1024, batchnorm=True)
			params = param_init_fflayer(params, _concat(prefix, 'o'), 1024, units, zero_init=True)

	return params

def synth_grad(tparams, prefix, inp):
	'''
	Synthetic gradient estimation using a linear model
	'''
	global args
	if args.sg_type == 'lin':
		return T.dot(inp, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')]
	
	elif args.sg_type == 'deep' or args.sg_type == 'lin_deep':
		outi = fflayer(tparams, inp, _concat(prefix, 'I'), nonlin='relu', batchnorm=True)
		outh = fflayer(tparams, outi, _concat(prefix,'H'), nonlin='relu', batchnorm=True)
		out1 = fflayer(tparams, outh + outi, _concat(prefix, '1'), nonlin='relu', batchnorm=True)
		out2 = fflayer(tparams, outh + out1, _concat(prefix, '2'), nonlin='relu', batchnorm=True)
		if args.sg_type == 'deep':
			return fflayer(tparams, outh + outi, _concat(prefix, 'o'), nonlin=None)
		elif args.sg_type == 'lin_deep':
			return T.dot(inp, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')] + fflayer(tparams, out2 + out1, _concat(prefix, 'o'), nonlin=None)
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

# no address provided for weights
if args.load is None or args.load is not None:
	# encoder
	params = param_init_fflayer(params, _concat(ff_e, 'i'), 14*28, 200)
	params = param_init_fflayer(params, _concat(ff_e, 'h'), 200, 100)

	# latent
	params = param_init_fflayer(params, _concat(ff_e, 'bern'), 100, latent_dim)
	
	# synthetic gradient module for the last encoder layer
	params = param_init_sgmod(params, _concat(sg, 'r'), latent_dim)

	# loss prediction neural network, conditioned on input and output (in this case the whole image), acts as the baseline
	if args.var_red == 'cmr':	
		params = param_init_fflayer(params, 'loss_pred', 28*28, 1)

	# decoder parameters
	params = param_init_fflayer(params, _concat(ff_d, 'n'), latent_dim, 100)
	params = param_init_fflayer(params, _concat(ff_d, 'h'), 100, 200)
	params = param_init_fflayer(params, _concat(ff_d, 'o'), 200, 14*28)

if args.load is not None:
	# restore from saved weights
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
	gt_unrepeated = train_gt[img_ids,:]
	
	# repeat args.repeat-times to compensate for the sampling process
	gt = T.extra_ops.repeat(gt_unrepeated, args.repeat, axis=0)

	# inputs for synthetic gradient networks
	target_gradients = T.matrix('tg', dtype='float32')
	activation = T.matrix('sg_input_probs', dtype='float32')
	# also provide the top half of the image as input the synthetic gradient subnetworks
	latent_gradients = T.matrix('sg_input_latgrads', dtype='float32')

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
out1 = fflayer(tparams, img, _concat(ff_e, 'i'))
out2 = fflayer(tparams, out1, _concat(ff_e,'h'))
out3 = fflayer(tparams, out2, _concat(ff_e, 'bern'), nonlin='sigmoid')

# repeat args.repeat-times so that for every input in a minibatch, there are args.repeat samples
latent_probs = T.extra_ops.repeat(out3, args.repeat, axis=0)

# sample a bernoulli distribution, which a binomial of 1 iteration
latent_samples = srng.binomial(size=latent_probs.shape, n=1, p=latent_probs, dtype=theano.config.floatX)

# decoding
outz = fflayer(tparams, latent_samples, _concat(ff_d, 'n'))
outh = fflayer(tparams, outz, _concat(ff_d, 'h'))
probs = fflayer(tparams, outh, _concat(ff_d, 'o'), nonlin='sigmoid')
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# Training
if args.mode == 'train':
	reconstruction_loss = T.nnet.binary_crossentropy(probs, gt).sum(axis=1)
	print "Computing synthetic gradients"

	# separate parameters for encoder, decoder and sg subnetworks
	param_dec = [val for key, val in tparams.iteritems() if 'ff_dec' in key]
	param_enc = [val for key, val in tparams.iteritems() if 'ff_enc' in key]
	param_sg = [val for key, val in tparams.iteritems() if 'sg' in key]

	print "Encoder parameters:", param_enc
	print "Decoder parameters:", param_dec
	print "Synthetic gradient subnetwork parameters:", param_sg
	
	print "Computing gradients wrt to decoder parameters"
	cost_decoder = T.mean(reconstruction_loss)
	grads_decoder = T.grad(cost_decoder, wrt=param_dec)
	
	# for better estimation, converts into a learnt straight through estimator
	gradz_unscaled = T.grad(cost_decoder, wrt=latent_samples)
	gradz = gradz_unscaled[:args.batch_size,:]
	for i in range(1, args.repeat):
		gradz += gradz_unscaled[i*args.batch_size: (i+1)*args.batch_size, :]
	gradz = gradz / args.repeat

	print "Computing gradients wrt to encoder parameters"
	# clipping for stability of gradients
	if args.clip_probs == 1:
		latent_probs_clipped = T.clip(latent_probs, 1e-7, 1-1e-7)
	elif args.clip_probs == 0:
		latent_probs_clipped = latent_probs
	
	# arguments to be considered constant when computing gradients
	consider_constant = [reconstruction_loss, latent_samples]

	if args.var_red is None:
		cost_encoder = T.mean(reconstruction_loss * -T.nnet.nnet.binary_crossentropy(latent_probs_clipped, latent_samples).sum(axis=1))
	
	elif args.var_red == 'mr':
		# unconditional mean is subtracted from the reconstruction loss, to yield a relatively lower variance unbiased REINFORCE estimator
		cost_encoder = T.mean((reconstruction_loss - T.mean(reconstruction_loss)) * -T.nnet.nnet.binary_crossentropy(latent_probs_clipped, latent_samples).sum(axis=1))
	
	elif args.var_red == 'cmr':
		# conditional mean is subtracted from the reconstruction loss to lower variance further
		baseline = T.extra_ops.repeat(fflayer(tparams, T.concatenate([img, gt_unrepeated], axis=1), 'loss_pred', nonlin='relu'), args.repeat, axis=0)
		cost_encoder = T.mean((reconstruction_loss - baseline.T) * -T.nnet.nnet.binary_crossentropy(latent_probs_clipped, latent_samples).sum(axis=1))

		# optimizing the predictor
		cost_pred = 0.5 * ((reconstruction_loss - baseline.T) ** 2).sum()
		
		params_loss_predictor = [val for key, val in tparams.iteritems() if 'loss_pred' in key]
		print "Loss predictor parameters:", params_loss_predictor

		grads_plp = T.grad(cost_pred, wrt=params_loss_predictor, consider_constant=[reconstruction_loss])
		consider_constant += [baseline]

	known_grads = OrderedDict()
	known_grads[out3] = synth_grad(tparams, _concat(sg, 'r'), T.concatenate([img, out3, gt_unrepeated, gradz], axis=1))
	grads_encoder = T.grad(None, wrt=param_enc, known_grads=known_grads)

	# combine grads in this order only
	if args.var_red == 'cmr':
		grads_net = grads_encoder + grads_plp + grads_decoder
	else:
		grads_net = grads_encoder + grads_decoder

	# computing target for synthetic gradient, will be output in every iteration
	sg_target = T.grad(cost_encoder, wrt=out3, consider_constant=consider_constant)
	
	loss_sg = 0.5 * ((target_gradients - synth_grad(tparams, _concat(sg, 'r'), T.concatenate([img, activation, gt_unrepeated, latent_gradients], axis=1))) ** 2).sum()
	grads_sg = T.grad(loss_sg, wrt=param_sg)

	cost = cost_decoder
	lr = T.scalar('lr', dtype='float32')

	inps_net = [img_ids]
	inps_sg = inps_net + [activation, target_gradients, latent_gradients]
	tparams_net = OrderedDict()
	tparams_sg = OrderedDict()
	for key, val in tparams.iteritems():
		print key
		if 'sg' in key:
			tparams_sg[key] = val
		else:
			tparams_net[key] = val

	print "Setting up optimizers"
	f_grad_shared, f_update = adam(lr, tparams_net, grads_net, inps_net, [cost, sg_target, out3, gradz])
	f_grad_shared_sg, f_update_sg = adam(lr, tparams_sg, grads_sg, inps_sg, loss_sg)
	
	print "Training"
	cost_report = open('./Results/' + args.latent_type + '/' + estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '.txt', 'w')
	id_order = range(len(trc))

	iters = 0
	min_cost = 100000.0
	epoch = 0
	condition = False
	if args.update_style == 'fixed':
		update_freq = args.repeat
	elif args.update_style == 'decay':
		update_freq = 1

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
			cost, t, ls, gradz = f_grad_shared(idlist)	
			f_update(args.learning_rate)
			cost_sg = 'NC'
			
			if iters % update_freq == 0 and not np.isnan((t**2).sum()):
				cost_sg = f_grad_shared_sg(idlist, ls, t, gradz)
				f_update_sg(args.learning_rate)
				epoch_cost_sg += cost_sg
			
			elif np.isnan((t**2).sum()):
				print "NaN encountered at", iters	
			
			# decay mode
			if args.update_style == 'decay':
				 if iters == 2000:
				 	update_freq = 2
				 elif iters == 5000:
				 	update_freq = 5
				 elif iters == 10000:
				 	update_freq = 10
				 elif iters == 20000:
				 	update_freq = 50
				 elif iters == 30000:
			 		update_freq = 100
			
			epoch_cost += cost
			min_cost = min(min_cost, cost)
			cost_report.write(str(epoch) + ',' + str(batch_id) + ',' + str(cost) + ',' + str(cost_sg) + ',' + str(time.time() - batch_start) + '\n')

		print ": Cost " + str(epoch_cost) + " : SG Cost " + str(epoch_cost_sg) + " : Time " + str(time.time() - epoch_start)
		
		# save every args.save_freq epochs
		if (epoch + 1) % args.save_freq == 0:
			print "Saving..."

			params = {}
			for key, val in tparams.iteritems():
				params[key] = val.get_value()

			# numpy saving
			np.savez('./Results/' + args.latent_type + '/' + estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(epoch+1) + '.npz', **params)
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
		np.savez('./Results/' + args.latent_type + '/' + estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(epoch) + '.npz', **params)
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
