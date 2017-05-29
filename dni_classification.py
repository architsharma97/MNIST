import sys

import numpy as np
from read_mnist import read, show
import theano
import theano.tensor as T
from utils import init_weights, _concat
from adam import adam

from collections import OrderedDict
import time

'''
Compare the training using DNIs (Synthetic Gradients) and normal Backpropagation.
The task is the classify MNIST digits using a simple 3-layer fully connected network.
1: Train = 0, Test = 1
2: Address for weights
'''

# meta-variables
batch_size = 100
learning_rate = 0.001

# regularization
lmbda = 0.0

# termination conditions: either max epochs ('e') or minimum loss levels for a minibatch ('c')
term_condition = 'e'
max_epochs = 100
minbatch_cost = 55.0
condition = False

# save every save_freq epochs
save_freq = 5

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
	elif nonlin == 'softplus':
		return T.nnet.nnet.softplus(T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')])

print "Getting data"
# collect training data and labels and does row major flattening
tri = np.asarray([img.flatten() for lbl, img in read(dataset='training', path ='MNIST/')], dtype=np.float32)
trl = np.asarray([lbl for lbl, img in read(dataset='training', path ='MNIST/')], dtype=np.int64)

# collect test data and label and does row major flattening
tei = np.asarray([img.flatten() for lbl, img in read(dataset='testing', path = 'MNIST/')], dtype=np.float32)
tel = np.asarray([lbl for lbl, img in read(dataset='testing', path = 'MNIST/')], dtype=np.int64)

print "Initializing parameters"

ff = 'ff'

# no address for weights 
if len(sys.argv) < 3:
	params = OrderedDict()

	params = param_init_fflayer(params, _concat(ff, '1'), 28*28, 300)
	params = param_init_fflayer(params, _concat(ff, '2'), 300, 150)
	params = param_init_fflayer(params, _concat(ff, 'o'), 150, 10)

else:
	params = np.load(sys.argv[2])

tparams = OrderedDict()
for key, val in params.iteritems():
	tparams[key] = theano.shared(val, name=key)

# Training graph
if len(sys.argv) < 2 or int(sys.argv[1]) == 0:
	print "Construncting the training graph"

	train_data = theano.shared(tri, name='train_data')
	train_labels = theano.shared(trl, name='train_labels')

	# pass a batch of indices
	img_ids = T.vector('ids', dtype='int64')
	img = train_data[img_ids, :]
	lbl = train_labels[img_ids, :]

# Test graph
else:
	print "Constructing the test graph"

	test_data = theano.shared(tei, name='test_data')
	test_labels = theano.shared(tel, name='test_labels')

	# pass a batch of indices
	img_ids = T.vector('ids', dtype='int64')
	img = test_data[img_ids, :]
	lbl = test_labels[img_ids, :]

out1 = fflayer(tparams, img, _concat(ff, '1'))
out2 = fflayer(tparams, out1, _concat(ff, '2'))
out3 = fflayer(tparams, out2, _concat(ff, 'o'), nonlin=None)
probs = T.nnet.nnet.softmax(out3)

# Training
if len(sys.argv) < 2 or int(sys.argv[1]) == 0:
	# cross-entropy loss: provide the index of 1 for a one-hot encoding
	loss = T.nnet.nnet.categorical_crossentropy(probs, lbl).sum()

	param_list = [val for key, val in tparams.iteritems()]
	
	# regularization
	weights_sum = 0.
	for val in param_list:
		weights_sum += (val**2).sum()
	loss += lmbda * weights_sum 

	grads = T.grad(loss, wrt=param_list)
	lr = T.scalar('learning_rate', dtype='float32')

	inps = [img_ids]
	print "Setting up optimizer"
	f_grad_shared, f_update = adam(lr, tparams, grads, inps, loss)

	print "Training"
	cost_report = open('./Results/classification/backprop/training_' + str(batch_size) + '_' + str(learning_rate) + '.txt', 'w')
	id_order = [i for i in range(len(tri))]

	min_cost = 100000.0
	epoch = 0
	while condition == False:
		print "Epoch " + str(epoch + 1),

		np.random.shuffle(id_order)
		epoch_cost = 0.
		epoch_start = time.time()
		for batch_id in range(len(tri)/batch_size):
			batch_start = time.time()

			idlist = id_order[batch_id*batch_size:(batch_id+1)*batch_size]
			cost = f_grad_shared(idlist)
			min_cost = min(min_cost, cost)

			f_update(learning_rate)

			epoch_cost += cost
			cost_report.write(str(epoch) + ',' + str(batch_id) + ',' + str(cost) + ',' + str(time.time() - batch_start) + '\n')

		print ": Cost " + str(epoch_cost) + " : Time " + str(time.time() - epoch_start)

		# save every save_freq epochs
		if (epoch + 1) % save_freq == 0:
			print "Saving..."

			params = {}
			for key, val in tparams.iteritems():
				params[key] = val.get_value()

			# numpy saving
			np.savez('./Results/classification/backprop/training_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(epoch+1) + '.npz', **params)
			print "Done!"

		epoch += 1
		if term_condition == 'c' and min_cost < minbatch_cost:
			condition = True
		elif term_condition == 'e' and epoch >= max_epochs:
			condition = True

	# saving the final model
	if epoch % save_freq != 0:
		print "Saving..."

		params = {}
		for key, val in tparams.iteritems():
			params[key] = val.get_value()

		# numpy saving
			np.savez('./Results/classification/backprop/training_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(epoch) + '.npz', **params)
		print "Done!"

else:
	pred = T.argmax(probs, axis=1)
	acc = T.mean(T.eq(pred, lbl)) * 100
	f = theano.function([img_ids], acc)

	# print accuracy over training data
	print f([i for i in range(len(tei))])
