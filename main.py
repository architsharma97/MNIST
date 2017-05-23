import sys

import numpy as np
from read_mnist import read, show
import theano
import theano.tensor as T
from utils import save_obj, load_obj, init_weights, _concat
from adam import adam

from collections import OrderedDict
import time

'''
Script Arguments
No arguments = train from random initialization
1: Train = 0, Test = 1
2: Address for weights
'''

seed = 42
learning_rate = 0.001
EPOCHS = 100
batch_size = 100
estimator = 'SF'
delta = 1e-12

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
	if nonlin == None:
		return T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')]
	elif nonlin == 'tanh':
		return T.tanh(T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')])
	elif nonlin == 'sigmoid':
		return T.nnet.nnet.sigmoid(T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')])

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
if len(sys.argv) < 3:
	params = OrderedDict()

	# encoder
	params = param_init_fflayer(params, _concat(ff_e, 'i'), 14*28, 200)
	params = param_init_fflayer(params, _concat(ff_e, 'h'), 200, 100)

	# latent distribution parameters
	params = param_init_fflayer(params, _concat(ff_e, 'mu'), 100, latent_dim)
	params = param_init_fflayer(params, _concat(ff_e, 'sd'), 100, latent_dim)

	# decoder parameters
	params = param_init_fflayer(params, _concat(ff_d, 'n'), latent_dim, 100)
	params = param_init_fflayer(params, _concat(ff_d, 'h'), 100, 200)
	params = param_init_fflayer(params, _concat(ff_d, 'o'), 200, 14*28)

else:
	# restore from saved weights
	params = np.load(sys.argv[2])

tparams = OrderedDict()
for key, val in params.iteritems():
	tparams[key] = theano.shared(val, name=key)

# Training graph
if len(sys.argv) < 2 or int(sys.argv[1]) == 0:
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

# Test graph
else:
	print "Contructing the test graph"
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
mu = fflayer(tparams, out2, _concat(ff_e, 'mu'), nonlin=None)
sd = fflayer(tparams, out2, _concat(ff_e, 'sd'), nonlin=None)

if "gpu" in theano.config.device:
	srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
else:
	srng = T.shared_randomstreams.RandomStreams(seed=seed)

# sampling from zero mean normal distribution
eps = srng.normal(mu.shape)
latent_samples = mu + sd * eps

# decoding
outz = fflayer(tparams, latent_samples, _concat(ff_d, 'n'))
outh = fflayer(tparams, outz, _concat(ff_d, 'h'))
probs = fflayer(tparams, outh, _concat(ff_d, 'o'), nonlin='sigmoid')

# Training
if len(sys.argv) < 2 or int(sys.argv[1]) == 0:
	
	reconstruction_loss = T.nnet.binary_crossentropy(probs, gt).sum(axis=1)

	# Uses the reparametrization trick
	if estimator == 'PD':
		print "Computing gradient estimators using PD"
		cost = T.mean(reconstruction_loss)
		param_list = [val for key, val in tparams.iteritems()]

		grads = T.grad(cost, wrt=param_list)

	if estimator == 'SF':
		print "Computing gradient estimators using REINFORCE"

		# separate parameters for encoder and decoder
		param_dec = [val for key, val in tparams.iteritems() if 'ff_dec' in key]
		param_enc = [val for key, val in tparams.iteritems() if 'ff_dec' not in key]
		print "Encoder parameters: ",
		print param_enc
		print "Decoder parameters: ",
		print param_dec

		print "Computing gradients wrt to decoder parameters"
		cost_decoder = T.mean(reconstruction_loss)
		grads_decoder = T.grad(cost_decoder, wrt=param_dec)

		print "Computing gradients wrt to encoder parameters"
		cost_encoder = T.mean(reconstruction_loss * (-0.5 * T.log(abs(sd) + delta).sum(axis=1) - 0.5 * (((latent_samples - mu)/sd) ** 2).sum(axis=1)))

		grads_encoder = T.grad(cost_encoder, wrt=param_enc, consider_constant=[reconstruction_loss])
		
		grads=grads_encoder + grads_decoder
	# learning rate
	lr = T.scalar('lr', dtype='float32')

	inps = [img_ids]

	print "Setting up optimizer"
	f_grad_shared, f_update = adam(lr, tparams, grads, inps, cost)

	print "Training"
	cost_report = open('./Results/' + estimator + '/training_' + estimator.lower() + '_' + str(batch_size) + '_' + str(learning_rate) + '.txt', 'w')
	id_order = [i for i in range(len(trc))]
	for epoch in range(EPOCHS):
		print "Epoch " + str(epoch + 1),

		np.random.shuffle(id_order)
		epoch_cost = 0.
		epoch_start = time.time()
		for batch_id in range(len(trc)/batch_size):
			batch_start = time.time()

			idlist = id_order[batch_id*batch_size:(batch_id+1)*batch_size]
			cost = f_grad_shared(idlist)
			f_update(learning_rate)

			epoch_cost += cost
			cost_report.write(str(epoch) + ',' + str(batch_id) + ',' + str(cost) + ',' + str(time.time() - batch_start) + '\n')
			

		print ": Cost " + str(epoch_cost) + " : Time " + str(time.time() - epoch_start)
		# save every 5 epochs
		if (epoch + 1) % 5 == 0:
			print "Saving..."

			params = {}
			for key, val in tparams.iteritems():
				params[key] = val.get_value()

			# numpy saving
			np.savez('./Results/' + estimator + '/training_' + estimator.lower() + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(epoch+1) + '.npz', **params)
			print "Done!"

# Test
else:
	# useful for one example at a time only
	prediction = probs > 0.5

	loss = abs(prediction-gt).sum()

	# compiling test function
	f = theano.function([img_ids], [prediction, loss])
	idx = 10
	pred, loss = f([idx])

	show(tec[idx].reshape(28,28))
	
	reconstructed_img = np.zeros((28*28,))
	reconstructed_img[:14*28] = tep[idx][0]
	reconstructed_img[14*28:] = pred
	show(reconstructed_img.reshape(28,28))
	print loss