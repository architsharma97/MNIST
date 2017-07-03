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
A model to complete the images from MNIST, when only the top half is given. Training is conducted using a new estimator for latent variables.
'''
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# training configuration
parser.add_argument('-m', '--mode', type=str, default='train', 
					help='train or test, note you still have to set the configure the latent type and estimator for proper testing.')
# while testing
parser.add_argument('-l', '--load', type=str, default=None, help='Path to weights')

# hyperparameters
parser.add_argument('-a', '--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('-e', '--z_corrate', type=float, default=0.0001, help='rate at which latent samples are corrected')
parser.add_argument('-b', '--batch_size', type=int, default=100, help='Size of the minibatch used for training')

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

code_name = args.base_code
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

params = OrderedDict()
# encoder
params = param_init_fflayer(params, _concat(ff_e, 'i'), 14*28, 200)
params = param_init_fflayer(params, _concat(ff_e, 'h'), 200, 100)

# latent
params = param_init_fflayer(params, _concat(ff_e, 'bern'), 100, latent_dim)

# decoder parameters
params = param_init_fflayer(params, _concat(ff_d, 'n'), latent_dim, 100)
params = param_init_fflayer(params, _concat(ff_d, 'h'), 100, 200)
params = param_init_fflayer(params, _concat(ff_d, 'o'), 200, 14*28)

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
	img = train[img_ids, :]
	gt = train_gt[img_ids, :]

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

latent_probs = fflayer(tparams, out2, _concat(ff_e, 'bern'), nonlin='sigmoid')

# sample a bernoulli distribution, which a binomial of 1 iteration
latent_samples = srng.binomial(size=latent_probs.shape, n=1, p=latent_probs, dtype=theano.config.floatX)

# decoding
outz = fflayer(tparams, latent_samples, _concat(ff_d, 'n'))
outh = fflayer(tparams, outz, _concat(ff_d, 'h'))
probs = fflayer(tparams, outh, _concat(ff_d, 'o'), nonlin='sigmoid')

if args.mode == 'train':

	reconstruction_loss = T.nnet.binary_crossentropy(probs, gt).sum(axis=1)

	# separate parameters for encoder and decoder
	param_dec = [val for key, val in tparams.iteritems() if 'ff_dec' in key]
	param_enc = [val for key, val in tparams.iteritems() if 'ff_enc' in key]
	print "Encoder parameters:", param_enc
	print "Decoder parameters: ", param_dec

	param_dec += [latent_samples]
	cost = T.mean(reconstruction_loss)
	grads_decoder = T.grad(cost, wrt=param_dec)
	gradz = grads_decoder[-1]
	grads_decoder = grads_decoder[:-1]

	z_corr = T.cast((latent_samples - args.z_corrate * gradz) > 0.5, 'float32')

	# for stability of gradients
	latent_probs_clipped = T.clip(latent_probs, 1e-7, 1-1e-7)
	cost_encoder = T.mean(-T.nnet.nnet.binary_crossentropy(latent_probs_clipped, z_corr).sum(axis=1))
	grads_encoder = T.grad(cost_encoder, wrt=param_enc, consider_constant=[z_corr])

	grads = grads_encoder + grads_decoder

	# learning rate
	lr = T.scalar('lr', dtype='float32')

	inps = [img_ids]

	print "Setting up optimizer"
	f_grad_shared, f_update = adam(lr, tparams, grads, inps, cost)

	print "Training"
	cost_report = open('./Results/disc/new/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '.txt', 'w')
	id_order = range(len(trc))

	iters = 0
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
			np.savez('./Results/disc/new/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(epoch+1) + '.npz', **params)
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
		np.savez('./Results/disc/new/training_' + code_name + '_' + str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(epoch) + '.npz', **params)
		print "Done!"

# Test
else:
	# useful for one example at a time only
	prediction = probs > 0.5

	loss = abs(prediction - gt).sum()

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