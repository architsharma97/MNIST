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

# 'backprop' or 'synthetic_gradients'
train_rou = 'synthetic_gradients'
code = 'sg_relu_reg_0.5'

# regularization
lmbda = 0.0

# termination conditions: either max epochs ('e') or minimum loss levels for a minibatch ('c')
term_condition = 'e'
max_epochs = 100
minbatch_cost = 55.0
condition = False

# save every save_freq epochs
save_freq = 25

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
	elif nonlin == 'relu':
		return T.nnet.nnet.relu(T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')])

def param_init_sgmod(params, prefix, units):
	'''
	Initializes a linear regression based model for estimating gradients, conditioned on the class labels
	'''
	params[_concat(prefix, 'W')] = init_weights(units, units, type_init='ortho')
	params[_concat(prefix, 'C')] = init_weights(10, units, type_init='ortho')
	params[_concat(prefix, 'b')] = np.zeros((units,)).astype('float32') 

	return params

def synth_grad(tparams, prefix, activation, labels_one_hot):
	'''
	Synthetic gradient estimation using a linear model
	'''

	return T.dot(activation, tparams[_concat(prefix, 'W')]) + T.dot(labels_one_hot, tparams[_concat(prefix, 'C')]) + tparams[_concat(prefix, 'b')]

print "Getting data"
# collect training data and labels and does row major flattening
tri = np.asarray([img.flatten() for lbl, img in read(dataset='training', path ='MNIST/')], dtype=np.float32)
trl = np.asarray([lbl for lbl, img in read(dataset='training', path ='MNIST/')], dtype=np.int64)

# collect test data and label and does row major flattening
tei = np.asarray([img.flatten() for lbl, img in read(dataset='testing', path = 'MNIST/')], dtype=np.float32)
tel = np.asarray([lbl for lbl, img in read(dataset='testing', path = 'MNIST/')], dtype=np.int64)

print "Initializing parameters"

ff = 'ff'
sg = 'sg'

# no address for weights 
if len(sys.argv) < 3:
	params = OrderedDict()

	params = param_init_fflayer(params, _concat(ff, '1'), 28*28, 300)
	params = param_init_fflayer(params, _concat(ff, '2'), 300, 150)

	if train_rou == 'synthetic_gradients':
		params = param_init_sgmod(params, _concat(sg, '1'), 300)
		params = param_init_sgmod(params, _concat(sg, '2'), 150)

	params = param_init_fflayer(params, _concat(ff, 'o'), 150, 10)

else:
	params = np.load(sys.argv[2])

tparams = OrderedDict()
for key, val in params.iteritems():
	tparams[key] = theano.shared(val, name=key)

# Training graph
if len(sys.argv) < 2 or int(sys.argv[1]) == 0:
	print "Constructing the training graph"

	train_data = theano.shared(tri, name='train_data')
	train_labels = theano.shared(trl, name='train_labels')

	# pass a batch of indices
	img_ids = T.vector('ids', dtype='int64')
	img = train_data[img_ids, :]
	lbl = train_labels[img_ids, :]
	if train_rou == 'synthetic_gradients':
		lbl_one_hot = T.extra_ops.to_one_hot(lbl, 10, dtype='float32')

# Test graph
else:
	print "Constructing the test graph"

	test_data = theano.shared(tei, name='test_data')
	test_labels = theano.shared(tel, name='test_labels')

	# pass a batch of indices
	img_ids = T.vector('ids', dtype='int64')
	img = test_data[img_ids, :]
	lbl = test_labels[img_ids, :]

out1 = fflayer(tparams, img, _concat(ff, '1'), nonlin='relu')
out2 = fflayer(tparams, out1, _concat(ff, '2'), nonlin='relu')
out3 = fflayer(tparams, out2, _concat(ff, 'o'), nonlin=None)
probs = T.nnet.nnet.softmax(out3)

# Training
if len(sys.argv) < 2 or int(sys.argv[1]) == 0:
	# cross-entropy loss: provide the index of 1 for a one-hot encoding
	loss = T.nnet.nnet.categorical_crossentropy(probs, lbl).sum()

	param_list = [val for key, val in tparams.iteritems() if 'sg' not in key]
	
	# # regularization
	# weights_sum = 0.
	# for val in param_list:
	# 	weights_sum += (val**2).sum()
	# loss += lmbda * weights_sum 

	if train_rou == 'backprop':
		grads = T.grad(loss, wrt=param_list)
	
	elif train_rou == 'synthetic_gradients':
		print "Computing synthetic gradients"
		
		# computing gradients for last layer
		var_list = [tparams['ff_o_W'], tparams['ff_o_b'], out2]
		grad_list_1 = T.grad(loss, wrt=var_list)

		known_grads = OrderedDict()

		var_list = [tparams['ff_2_W'], tparams['ff_2_b'], out1]
		known_grads[out2] = synth_grad(tparams, _concat(sg, '2'), out2, lbl_one_hot)
		grad_list_2 = T.grad(loss, wrt=var_list, known_grads=known_grads)
		
		var_list = [tparams['ff_1_W'], tparams['ff_1_b']]
		known_grads[out1] = synth_grad(tparams, _concat(sg, '1'), out1, lbl_one_hot)
		grad_list_3 = T.grad(loss, wrt=var_list, known_grads=known_grads)
		
		# define a loss for synthetic gradient modules
		loss_sg = 0.5 * ((grad_list_1[2] - known_grads[out2]) ** 2).sum() + 0.5 * ((grad_list_2[2] - known_grads[out1]) ** 2).sum()
		sg_params_list = [val for key, val in tparams.iteritems() if 'sg' in key]

		grads_sg = T.grad(loss_sg, wrt=sg_params_list, consider_constant=[grad_list_1[2], grad_list_2[2], out1, out2])
		grads_net = grad_list_3 + grad_list_2[:2] + grad_list_1[:2]
	
	lr = T.scalar('learning_rate', dtype='float32')

	inps = [img_ids]
	print "Setting up optimizer"
	
	if train_rou == 'backprop':
		f_grad_shared, f_update = adam(lr, tparams, grads, inps, loss)
	
	elif train_rou == 'synthetic_gradients':
		
		# split params for sg modules and network
		tparams_net = OrderedDict()
		tparams_sg = OrderedDict()
		for key, val in tparams.iteritems():
			if 'sg' in key:
				tparams_sg[key] = tparams[key]
			else:
				tparams_net[key] = tparams[key]
		# check shapes of gradients
		f_grad_check = theano.function(inps, grads_net + grads_sg)
		comp_grads = f_grad_check(range(100))

		i = 0
		for key, val in tparams_net.iteritems():
			print key, comp_grads[i].shape
			i += 1

		for key, val in tparams_sg.iteritems():
			print key, comp_grads[i].shape
			i += 1

		f_grad_shared, f_update = adam(lr, tparams_net, grads_net, inps, loss)
		f_grad_shared_sg, f_update_sg = adam(lr, tparams_sg, grads_sg, inps, loss_sg)

	print "Training"
	cost_report = open('./Results/classification/' + train_rou + '/training_' + code + '_' + str(batch_size) + '_' + str(learning_rate) + '.txt', 'w')
	id_order = range(len(tri))

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

			if train_rou == 'synthetic_gradients':
				cost_sg = f_grad_shared_sg(idlist)
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
			np.savez('./Results/classification/' + train_rou + '/training_' + code + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(epoch+1) + '.npz', **params)
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
			np.savez('./Results/classification/' + train_rou + '/training_' + code + '_' + str(batch_size) + '_' + str(learning_rate) + '_' + str(epoch) + '.npz', **params)
		print "Done!"

else:
	pred = T.argmax(probs, axis=1)
	acc = T.mean(T.eq(pred, lbl)) * 100
	f = theano.function([img_ids], acc)

	# print accuracy over training data
	print f([i for i in range(len(tei))])