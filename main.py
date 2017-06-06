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
					help='Type of estimator to be used for the stochastic node. Choose REINFORCE (SF) or Path Derivative (PD)')
parser.add_argument('-o', '--latent_type', type=str, default='disc', 
					help='Either discrete bernoulli (disc) or continous gaussian (cont)')

# using REINFORCE
parser.add_argument('-r', '--repeat', type=int, default=1, 
					help='Determines the number of samples per training example for SF estimator, not to be provided with PD estimator')

# gumbel-softmax configuration which is used for PD estimators with discrete latent variables
parser.add_argument('-g', '--sample_style', type=int, default=0,
					help='Gumbel-softmax sampling can be followed up by hard sampling (1). It would ensure that the sampled vector contains 1s and 0s.')
# while testing
parser.add_argument('-l', '--load', type=str, default=None, help='Path to weights')

# hyperparameters
parser.add_argument('-a', '--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=100, help='Size of the minibatch used for training')
parser.add_argument('-c', '--regularization', type=float, default=0., help='Regularization constant')



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
parser.add_argument('-p', '--clip_probs', type=int, default=1,
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

def param_init_fflayer(params, prefix, nin, nout):
	'''
	Initializes weights for a feedforward layer
	'''
	params[_concat(prefix,'W')] = init_weights(nin, nout, type_init='ortho')
	params[_concat(prefix,'b')] = np.zeros((nout,)).astype('float32')

	return params

def fflayer(tparams, state_below, prefix, nonlin='tanh'):
	'''
	A feedforward layer
	'''
	preact = T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')]
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
latent_dim = 50

# no address provided for weights
if args.load is None:
	params = OrderedDict()

	# encoder
	params = param_init_fflayer(params, _concat(ff_e, 'i'), 14*28, 200)
	params = param_init_fflayer(params, _concat(ff_e, 'h'), 200, 100)

	# latent distribution parameters
	if args.latent_type == 'cont':
		params = param_init_fflayer(params, _concat(ff_e, 'mu'), 100, latent_dim)
		params = param_init_fflayer(params, _concat(ff_e, 'sd'), 100, latent_dim)
	
	elif args.latent_type == 'disc':
		params = param_init_fflayer(params, _concat(ff_e, 'bern'), 100, latent_dim)

	# decoder parameters
	params = param_init_fflayer(params, _concat(ff_d, 'n'), latent_dim, 100)
	params = param_init_fflayer(params, _concat(ff_d, 'h'), 100, 200)
	params = param_init_fflayer(params, _concat(ff_d, 'o'), 200, 14*28)

else:
	# restore from saved weights
	params = np.load(args.load)

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
	gt = train_gt[img_ids,:]
	if args.estimator == 'SF':
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
out1 = fflayer(tparams, img, _concat(ff_e, 'i'))
out2 = fflayer(tparams, out1, _concat(ff_e,'h'))

# latent parameters
if args.latent_type == 'cont':
	mu = fflayer(tparams, out2, _concat(ff_e, 'mu'), nonlin=None)
	sd = fflayer(tparams, out2, _concat(ff_e, 'sd'), nonlin='softplus')

	# sampling from zero mean normal distribution
	eps = srng.normal(mu.shape)
	latent_samples = mu + sd * eps

elif args.latent_type == 'disc':
	
	latent_probs = fflayer(tparams, out2, _concat(ff_e, 'bern'), nonlin='sigmoid')

	if args.estimator == 'SF':
		latent_probs = T.extra_ops.repeat(latent_probs, args.repeat, axis=0)

		# sample a bernoulli distribution, which a binomial of 1 iteration
		latent_samples = srng.binomial(size=latent_probs.shape, n=1, p=latent_probs, dtype=theano.config.floatX)
	
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
			dummy = latent_samples_soft[1,:,:] > 0.5 - latent_samples_soft[1, :, :]
			latent_samples = latent_samples_soft[1,:,:] + dummy
		else:
			latent_samples = latent_samples_soft[1,:,:]

# decoding
outz = fflayer(tparams, latent_samples, _concat(ff_d, 'n'))
outh = fflayer(tparams, outz, _concat(ff_d, 'h'))
probs = fflayer(tparams, outh, _concat(ff_d, 'o'), nonlin='sigmoid')
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

# Training
if args.mode == 'train':

	reconstruction_loss = T.nnet.binary_crossentropy(probs, gt).sum(axis=1)

	# Uses the reparametrization trick
	if args.estimator == 'PD':
		print "Computing gradient estimators using PD"
		cost = T.mean(reconstruction_loss)
		param_list = [val for key, val in tparams.iteritems()]
		
		if args.latent_type == 'disc' and args.sample_style == 1:
			# equivalent to stop_gradient trick in tensorflow
			grads = T.grad(cost, wrt=param_list, consider_constant=[dummy])
		
		elif args.latent_type == 'cont':
			grads = T.grad(cost, wrt=param_list)

	if args.estimator == 'SF':
		print "Computing gradient estimators using REINFORCE"

		# separate parameters for encoder and decoder
		param_dec = [val for key, val in tparams.iteritems() if 'ff_dec' in key]
		param_enc = [val for key, val in tparams.iteritems() if 'ff_dec' not in key]
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
			# for stability of gradients
			latent_probs_clipped = T.clip(latent_probs, 1e-7, 1-1e-7)
			cost_encoder = T.mean((reconstruction_loss - T.mean(reconstruction_loss)) * -T.nnet.nnet.binary_crossentropy(latent_probs_clipped, latent_samples).sum(axis=1))

		# regularization
		weights_sum_enc = 0.
		for val in param_enc:
			weights_sum_enc += (val**2).sum()
		cost_encoder += args.regularization * weights_sum_enc

		grads_encoder = T.grad(cost_encoder, wrt=param_enc, consider_constant=[reconstruction_loss, latent_samples])

		# combine grads in this order only
		grads = grads_encoder + grads_decoder

		cost = cost_decoder

	# learning rate
	lr = T.scalar('lr', dtype='float32')

	inps = [img_ids]
	if args.estimator == 'PD' and args.latent_type == 'disc':
		inps += [temperature]
		temperature_min = temperature_init/2.0
		anneal_rate = 0.00003

	print "Setting up optimizer"
	f_grad_shared, f_update = adam(lr, tparams, grads, inps, cost)

	print "Training"
	cost_report = open('./Results/' + args.latent_type + '/' + args.estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '.txt', 'w')
	id_order = range(len(trc))

	iters = 0
	cur_temp = temperature_init
	min_cost = 100000.0
	epoch = 0
	condition = False

	while condition == False:
		print "Epoch " + str(epoch + 1),

		np.random.shuffle(id_order)
		epoch_cost = 0.
		epoch_start = time.time()
		for batch_id in range(len(trc)/args.batch_size):
			batch_start = time.time()

			idlist = id_order[batch_id*args.batch_size:(batch_id+1)*args.batch_size]
			if args.estimator == 'PD' and args.latent_type == 'disc':
				# fprint(idlist, cur_temp)
				cost = f_grad_shared(idlist, cur_temp)
				iters += 1
				if iters % 1000 == 0:
					cur_temp = np.maximum(temperature_init*np.exp(-anneal_rate*iters, dtype=np.float32), temperature_min)
			else:
				# fprint(idlist)
				cost = f_grad_shared(idlist)	
				min_cost = min(min_cost, cost)

			f_update(args.learning_rate)

			epoch_cost += cost
			cost_report.write(str(epoch) + ',' + str(batch_id) + ',' + str(cost) + ',' + str(time.time() - batch_start) + '\n')

		print ": Cost " + str(epoch_cost) + " : Time " + str(time.time() - epoch_start)
		
		# save every args.save_freq epochs
		if (epoch + 1) % args.save_freq == 0:
			print "Saving..."

			params = {}
			for key, val in tparams.iteritems():
				params[key] = val.get_value()

			# numpy saving
			np.savez('./Results/' + args.latent_type + '/' + args.estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(epoch+1) + '.npz', **params)
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
		np.savez('./Results/' + args.latent_type + '/' + args.estimator + '/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(epoch) + '.npz', **params)
		print "Done!"

# Test
else:
	# useful for one example at a time only
	prediction = probs > 0.5

	loss = abs(prediction - gt).sum()

	# compiling test function
	inps = [img_ids]
	if args.estimator == 'PD' and args.latent_type =='disc':
		inps += [temperature]

	f = theano.function(inps, [prediction, loss])
	idx = 10
	if args.estimator == 'PD' and args.latent_type =='disc':
		pred, loss = f([idx], 0.5)
	else:
		pred, loss = f([idx])

	show(tec[idx].reshape(28,28))

	reconstructed_img = np.zeros((28*28,))
	reconstructed_img[:14*28] = tep[idx][0]
	reconstructed_img[14*28:] = pred
	show(reconstructed_img.reshape(28,28))
	print loss