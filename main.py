import sys

import numpy as np
from read_mnist import read, show
import theano
import theano.tensor as T
from utils import init_weights, _concat
from adam import adam
from theano.compile.nanguardmode import NanGuardMode
import argparse

from collections import OrderedDict
import time

'''
A model to complete the images from MNIST, when only the top half is given. The top half is encoded into a latent distribution, 
from which a sample is generated to be decoded. The latent variable can be continuous or discrete. For continuous, PD (-> reparametrization) and SF estimators are wrt to 
gaussian latent variable. For discrete, SF estimators are wrt bernoulli latent variable, and PD estimators wrt to gumbel-softmax distribution (-> reparametrization).
Gumbel-softmax can be used in either hard or soft sampling mode, hard sampling mode converts to one-hot vector.
'''
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training configuration
parser.add_argument('-m', '--mode', type=str, default='train', 
					help='train or test, note you still have to set the configure the latent type and estimator for proper testing.')
parser.add_argument('-e','--estimator', type=str, default='SF',
					help='Type of estimator to be used for the stochastic node. Choose REINFORCE (SF), Path Derivative (PD) or Straight Through (ST) with discrete.')
parser.add_argument('-o', '--latent_type', type=str, default='disc', 
					help='Either discrete bernoulli (disc) or continous gaussian (cont)')
parser.add_argument('-z', '--bn_type', type=int, default=1,
					help='0: BN->Matrix Multiplication->Nonlinearity, 1: Matrix Multiplication->BN->Nonlinearity')

# using REINFORCE or ST
parser.add_argument('-r', '--repeat', type=int, default=1, 
					help='Determines the number of samples per training example for SF estimator, not to be provided with PD estimator')
# using REINFORCE
parser.add_argument('-v', '--var_red', type=str, default='cmr',
					help='Use different control variates, unconditional mean (mr) and conditional mean (cmr)')
parser.add_argument('-u', '--exptemp', type=float, default=100.0,
					help='Rewards are exponentiated: This controls the temperature.')

# gumbel-softmax configuration which is used for PD estimators with discrete latent variables
parser.add_argument('-g', '--sample_style', type=int, default=0,
					help='Gumbel-softmax sampling can be followed up by hard sampling (1). It would ensure that the sampled vector contains 1s and 0s.')
# while testing
parser.add_argument('-l', '--load', type=str, default=None, help='Path to weights')
parser.add_argument('-aa', '--val_file', type=str, default=None, help='File where validation data is written')

# hyperparameters
parser.add_argument('-a', '--learning_rate', type=float, default=0.0002, help='Learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=100, help='Size of the minibatch used for training')
parser.add_argument('-c', '--regularization', type=float, default=0., help='Regularization constant')
parser.add_argument('-ab', '--slash_rate', type=float, default=1.0, 
					help='Factor by which learning rate is reduced every 20 epochs. No reduction by default')

# termination of training
parser.add_argument('-t', '--term_condition', type=str, default='epochs', 
					help='Training terminates either when number of epochs are completed (epochs) or when minimum cost is achieved for a batch (mincost)')
parser.add_argument('-n', '--num_epochs', type=int, default=1000, 
					help='Number of epochs, to be specified when termination condition is epochs')
parser.add_argument('-d', '--min_cost', type=float, default=55.0, 
					help='Minimum cost to be achieved for a minibatch, to be specified when termination condition is mincost')

# saving
parser.add_argument('-s', '--save_freq', type=int, default=100, 
					help='Number of epochs after which weights should be saved')
parser.add_argument('-f', '--base_code', type=str, default='',
					help='A unique identifier for saving purposes')

# miscellaneous
parser.add_argument('-p', '--clip_probs', type=int, default=0,
					help='clip latent probabilities (1) or not (0), useful for testing training under NaNs')
parser.add_argument('-q', '--random_seed', type=int, default=42, help='Seed to initialize random streams')

args = parser.parse_args()

# random seed and initialization of stream
if "gpu" in theano.config.device:
	srng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed=args.random_seed)
else:
	srng = T.shared_randomstreams.RandomStreams(seed=args.random_seed)

# used for parameter saving and cost reports
if args.estimator == 'PD':
	code_name = args.base_code
else:
	code_name = args.base_code + '_' + str(args.repeat) 

# for numerical stability
delta = 1e-10
init_rate = args.learning_rate
temperature_init = 1.0
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# converts images into binary images for simplicity
def binarize_img(img):
	return np.asarray(img >= 100, dtype=np.int8)

# split into two images
def split_img(img):
	# images are flattened into a vector: just need to split into half
	veclen = len(img)
	return (img[:veclen/2], img[veclen/2:])

def param_init_fflayer(params, prefix, nin, nout, zero_init=False, batchnorm=False, skip_running_vars=False):
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
		
		# it is not necessary for deep synthetic subnetworks to track running averages as they are not used in test time
		if not skip_running_vars:
			params[_concat(prefix, 'rm')] = np.zeros((1, dim)).astype('float32')
			params[_concat(prefix, 'rv')] = np.ones((1, dim), dtype=np.float32)

	return params

def fflayer(tparams, state_below, prefix, nonlin='tanh', batchnorm=None, dropout=None, skip_running_vars=False):
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

		# shared variable updates
		if not skip_running_vars:
			# define new variables which will be used to update the shared variables
			tparams[_concat(prefix, 'rmu')] = tparams[_concat(prefix, 'rm')] * (1 - running_average_factor) + mean * running_average_factor
			tparams[_concat(prefix, 'rvu')] = tparams[_concat(prefix, 'rv')] * (1 - running_average_factor) + (m / (m - 1)) * var * running_average_factor
		
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
latent_dim = 100

params = OrderedDict()

# encoder
params = param_init_fflayer(params, _concat(ff_e, 'i'), 14*28, 200, batchnorm=True)
params = param_init_fflayer(params, _concat(ff_e, 'h'), 200, 100, batchnorm=True)

# latent distribution parameters
if args.latent_type == 'cont':
	if args.bn_type == 0:
		params = param_init_fflayer(params, _concat(ff_e, 'mu'), 100, latent_dim, batchnorm=True)
		params = param_init_fflayer(params, _concat(ff_e, 'sd'), 100, latent_dim, batchnorm=True)
	else:
		params = param_init_fflayer(params, _concat(ff_e, 'mu'), 100, latent_dim, batchnorm=False)
		params = param_init_fflayer(params, _concat(ff_e, 'sd'), 100, latent_dim, batchnorm=False)

elif args.latent_type == 'disc':
	if args.bn_type == 0:
		params = param_init_fflayer(params, _concat(ff_e, 'bern'), 100, latent_dim, batchnorm=True)
	else:
		params = param_init_fflayer(params, _concat(ff_e, 'bern'), 100, latent_dim, batchnorm=False)
	
	if args.estimator == 'SF' and args.var_red == 'cmr':
		# loss prediction neural network, conditioned on input and output (in this case the whole image). Acts as the baseline
		params = param_init_fflayer(params, 'loss_pred', 28*28, 1)

# decoder parameters
params = param_init_fflayer(params, _concat(ff_d, 'n'), latent_dim, 100)
params = param_init_fflayer(params, _concat(ff_d, 'h'), 100, 200, batchnorm=True)
if args.bn_type == 0 :
	params = param_init_fflayer(params, _concat(ff_d, 'o'), 200, 14*28, batchnorm=True)
else:
	params = param_init_fflayer(params, _concat(ff_d, 'o'), 200, 14*28, batchnorm=False)

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
	img = train[img_ids, :]
	gt = train_gt[img_ids, :]
	if args.estimator == 'SF' or args.estimator == 'ST':
		gt = T.extra_ops.repeat(gt, args.repeat, axis=0)

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
out2 = fflayer(tparams, out1, _concat(ff_e,'h'), batchnorm=args.mode)

# latent parameters
if args.latent_type == 'cont':
	mu = fflayer(tparams, out2, _concat(ff_e, 'mu'), nonlin=None)
	sd = fflayer(tparams, out2, _concat(ff_e, 'sd'), nonlin='softplus')

	# sampling from zero mean normal distribution
	eps = srng.normal(mu.shape)
	latent_samples = mu + sd * eps

elif args.latent_type == 'disc':
	if args.bn_type == 0:
		latent_probs = fflayer(tparams, out2, _concat(ff_e, 'bern'), nonlin='sigmoid', batchnorm=args.mode)
	else:
		latent_probs = fflayer(tparams, out2, _concat(ff_e, 'bern'), nonlin='sigmoid', batchnorm=None)

	if args.estimator == 'SF':
		# clipped for stability of gradients
		if args.clip_probs:
			latent_probs_r = T.clip(T.extra_ops.repeat(latent_probs, args.repeat, axis=0), 1e-7, 1-1e-7)
		else:
			latent_probs_r = T.extra_ops.repeat(latent_probs, args.repeat, axis=0)

		# sample a bernoulli distribution, which a binomial of 1 iteration
		latent_samples = srng.binomial(size=latent_probs_r.shape, n=1, p=latent_probs_r, dtype=theano.config.floatX)
	
	elif args.estimator == 'PD':
		# sample a gumbel-softmax distribution
		temperature = T.scalar('temp', dtype='float32')
		latent_probs_c = 1. - latent_probs
		prob_vector = T.stack([latent_probs_c, latent_probs])
		gumbel_samples = -T.log(-T.log(srng.uniform(prob_vector.shape, low=0.0, high=1.0, dtype='float32') + delta) + delta)

		latent_samples_unnormalized = ((T.log(prob_vector + delta) + gumbel_samples)/temperature)
		
		# custom softmax for tensors
		e_x = T.exp(latent_samples_unnormalized - latent_samples_unnormalized.max(axis=0, keepdims=True))
		latent_samples_soft = e_x / e_x.sum(axis=0, keepdims=True)
		
		if args.sample_style == 1:
			dummy = latent_samples_soft[1,:,:] > 0.5 - latent_samples_soft[1,:,:]
			latent_samples = latent_samples_soft[1,:,:] + dummy
		else:
			latent_samples = latent_samples_soft[1,:,:]
	
	# straight through estimator
	elif args.estimator == 'ST':
		latent_probs = T.extra_ops.repeat(latent_probs, args.repeat, axis=0)

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

	reconstruction_loss = T.nnet.binary_crossentropy(probs, gt).mean(axis=1)

	# Uses the reparametrization trick
	if args.estimator == 'PD':
		print "Computing gradient estimators using PD"
		cost = T.mean(reconstruction_loss)
		param_list = [val for key, val in tparams.iteritems() if ('rm' not in key and 'rv' not in key)]
		
		if args.latent_type == 'disc' and args.sample_style == 1:
			# equivalent to stop_gradient trick in tensorflow
			grads = T.grad(cost, wrt=param_list, consider_constant=[dummy])
		
		elif args.latent_type == 'cont':
			grads = T.grad(cost, wrt=param_list + [mu])
			xtranorm = T.mean(grads[-1] ** 2)
			grads = grads[:-1]

	if args.estimator == 'SF':
		print "Computing gradient estimators using REINFORCE"

		# separate parameters for encoder and decoder
		param_dec = [val for key, val in tparams.iteritems() if ('ff_dec' in key) and ('rm' not in key and 'rv' not in key)]
		param_enc = [val for key, val in tparams.iteritems() if ('ff_enc' in key) and ('rm' not in key and 'rv' not in key)]
		print "Encoder parameters:", param_enc
		print "Decoder parameters: ", param_dec

		print "Computing gradients wrt to decoder parameters"
		cost_decoder = T.mean(reconstruction_loss)

		# regularization
		weights_sum_dec = 0.
		for val in param_dec:
			weights_sum_dec += (val**2).sum()
		cost_decoder += args.regularization * weights_sum_dec

		grads_decoder = T.grad(cost_decoder, wrt=param_dec)

		print "Computing gradients wrt to encoder parameters"
		if args.latent_type == 'cont':
			cost_encoder = T.mean(reconstruction_loss * (-0.5 * T.log(abs(sd) + delta).sum(axis=1) - 0.5 * (((latent_samples - mu)/(sd + delta)) ** 2).sum(axis=1)))
			
		elif args.latent_type =='disc':
			# arguments to be considered constant when computing gradients
			consider_constant = [reconstruction_loss, latent_samples]

			if args.var_red is None:
				cost_encoder = T.mean(reconstruction_loss * T.switch(latent_samples, T.log(latent_probs_r), T.log(1. - latent_probs_r)).sum(axis=1))
				
			elif args.var_red == 'mr':
				# unconditional mean is subtracted from the reconstruction loss, to yield a relatively lower variance unbiased REINFORCE estimator
				cost_encoder = T.mean((reconstruction_loss - T.mean(reconstruction_loss)) * T.switch(latent_samples, T.log(latent_probs_r), T.log(1. - latent_probs_r)).sum(axis=1))
			
			elif args.var_red == 'cmr':
				# conditional mean is subtracted from the reconstruction loss to lower variance further
				baseline = T.extra_ops.repeat(fflayer(tparams, T.concatenate([img, train_gt[img_ids, :]], axis=1), 'loss_pred', nonlin='relu'), args.repeat, axis=0)
				cost_encoder = T.mean((reconstruction_loss - baseline.T) * T.switch(latent_samples, T.log(latent_probs_r), T.log(1. - latent_probs_r)).sum(axis=1))

				# optimizing the predictor
				cost_pred = T.mean((reconstruction_loss - baseline.T) ** 2)
				
				params_loss_predictor = [val for key, val in tparams.iteritems() if 'loss_pred' in key]
				print "Loss predictor parameters:", params_loss_predictor

				grads_plp = T.grad(cost_pred, wrt=params_loss_predictor, consider_constant=[reconstruction_loss])
				consider_constant += [baseline]
		
		# regularization
		weights_sum_enc = 0.
		for val in param_enc:
			weights_sum_enc += (val**2).sum()
		cost_encoder += args.regularization * weights_sum_enc

		grads_encoder = T.grad(cost_encoder, wrt=param_enc + [latent_probs], consider_constant=consider_constant)
		xtranorm = T.mean(grads_encoder[-1] ** 2)
		grads_encoder = grads_encoder[:-1]
		
		# combine grads in this order only
		if args.estimator == 'SF' and args.var_red == 'cmr':
			grads = grads_encoder + grads_plp + grads_decoder
		else:
			grads = grads_encoder + grads_decoder

		cost = cost_decoder


	if args.estimator == 'ST':
		print "Computing gradients using ST"
		cost = T.mean(reconstruction_loss)
		param_list = [val for key, val in tparams.iteritems() if ('rm' not in key) and ('rv' not in key)]

		if args.latent_type =='disc':
			# equivalent to stop_gradient trick in tensorflow
			grads = T.grad(cost, wrt=param_list + [latent_probs], consider_constant=[dummy])
			xtranorm = T.mean(grads[-1] ** 2)
			grads = grads[:-1]
			
		elif args.latent_type == 'cont':
			print "Nothing defined for this state"

	# learning rate
	lr = T.scalar('lr', dtype='float32')

	inps = [img_ids]
	if args.estimator == 'PD' and args.latent_type == 'disc':
		inps += [temperature]
		temperature_min = temperature_init/2.0
		anneal_rate = 0.00003

	tparams_net = OrderedDict()
	updates_bn = []
	for key, val in tparams.iteritems():
		if ('rmu' in key) or ('rvu' in key):
			continue
		elif 'rm' in key or 'rv' in key:
			updates_bn.append((tparams[key], tparams[key + 'u']))
		else:
			tparams_net[key] = val
	
	print "Setting up optimizer"
	f_grad_shared, f_update = adam(lr, tparams_net, grads, inps, [cost, xtranorm], ups=updates_bn)

	print "Training"
	cost_report = open('./Results/' + args.latent_type + '/' + args.estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '.txt', 'w')
	id_order = range(len(trc))

	iters = 0
	cur_temp = temperature_init
	min_cost = 100000.0
	epoch = 0
	condition = False

	while condition == False:
		if iters != 0 and iters % (20 * 600) == 0 and args.learning_rate > 1e-7:
			args.learning_rate /= args.slash_rate
			print "Updated main network learning rate:", args.learning_rate

		print "Epoch " + str(epoch + 1),

		np.random.shuffle(id_order)
		epoch_cost = 0.
		epoch_start = time.time()
		for batch_id in range(len(trc)/args.batch_size):
			batch_start = time.time()
			iters += 1

			idlist = id_order[batch_id*args.batch_size:(batch_id+1)*args.batch_size]
			if args.estimator == 'PD' and args.latent_type == 'disc':
				# fprint(idlist, cur_temp)
				cost = f_grad_shared(idlist, cur_temp)
				if iters % 1000 == 0:
					cur_temp = np.maximum(temperature_init*np.exp(-anneal_rate*iters, dtype=np.float32), temperature_min)
			else:
				# fprint(idlist)
				cost, xtra = f_grad_shared(idlist)	
				min_cost = min(min_cost, cost)
			
			f_update(args.learning_rate)

			epoch_cost += cost
			cost_report.write(str(epoch) + ',' + str(batch_id) + ',' + str(cost) + ',' + str(xtra) + ',' + str(time.time() - batch_start) + '\n')

		print ": Cost " + str(epoch_cost) + " : Time " + str(time.time() - epoch_start)
		
		# save every args.save_freq epochs
		if (epoch + 1) % args.save_freq == 0:
			print "Saving...",

			params = {}
			for key, val in tparams.iteritems():
				if not (('rmu' in key) or ('rvu' in key)):
					params[key] = val.get_value()

			# numpy saving
			np.savez('./Results/' + args.latent_type + '/' + args.estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(init_rate) + '_' + str(epoch+1) + '.npz', **params)
			print "Done!"

		epoch += 1
		if args.term_condition == 'mincost' and min_cost < args.min_cost:
			condition = True
		elif args.term_condition == 'epochs' and epoch >= args.num_epochs:
			condition = True
	
	# saving the final model
	if epoch % args.save_freq != 0:
		print "Saving...",

		for key, val in tparams.iteritems():
				if not (('rmu' in key) or ('rvu' in key)):
					params[key] = val.get_value()

		# numpy saving
		np.savez('./Results/' + args.latent_type + '/' + args.estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(init_rate) + '_' + str(epoch) + '.npz', **params)
		print "Done!"

# Test
else:
	# useful for one example at a time only
	loss = T.mean(T.nnet.binary_crossentropy(probs, gt))

	# compiling test function
	inps = [img_ids]
	if args.estimator == 'PD' and args.latent_type =='disc':
		inps += [temperature]

	f = theano.function(inps, [loss])
	if args.estimator == 'PD' and args.latent_type =='disc':
		loss = f(range(len(tec)), 0.5)
	else:
		loss = f(range(len(tec)))

	# show(tec[idx].reshape(28,28))

	# reconstructed_img = np.zeros((28*28,))
	# reconstructed_img[:14*28] = tep[idx][0]
	# reconstructed_img[14*28:] = pred
	# show(reconstructed_img.reshape(28,28))
	if args.val_file is None:
		print loss
	else:
		val_report = open(args.val_file, 'a')
		val_report.write(str(loss[0]) + '\n')