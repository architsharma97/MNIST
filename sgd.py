import numpy
import theano
import theano.tensor as T

class SGD():
    def __init__(self,  lr=0.01, momentum=0.9, nesterov=False, **kargs):
        self.updates = []
        self.constraint_updates = []
        self.alpha = numpy.float32(0.9)
        for item in kargs:
            self.__dict__[item] = kargs[item]

        self.momentum = theano.shared(value=numpy.float32(momentum), name='momentum', borrow=True) 
        self.lr = theano.shared(value=numpy.float32(lr), name='lr', borrow=True)
        self.nesterov = nesterov

    

    def get_gradients(self, total_loss, params):
        grads = T.grad(total_loss, params)

        if hasattr(self, 'clip_norm') and self.clip_norm >= 0:
            norm = [T.sqrt(T.sum(g**2)) for g in grads]
            grads = [T.switch(T.ge(n,  self.clip_norm),  self.clip_norm*g/n, g) for g,n in zip(grads, norm)]

        if hasattr(self, 'clip_value') and self.clip_value >= 0:
            grads = [T.clip(g, -self.clip_value, self.clip_value) for g in grads]

        if hasattr(self, 'clip_function'):
            grads = [self.clip_activation(g) for g in grads]
        
        return grads

    def get_grad_updates(self, total_loss=None, params=None):
        '''
        total_loss is a scalar
        params is a list of parameters
        '''
        updates = []
        gparams = self.get_gradients(total_loss, params)
        
        # momentum params
        for param, gparam in zip(params, gparams):
            val = numpy.zeros((param.get_value(borrow=True).shape),dtype='float32')
            if self.momentum.get_value() > 0:
                mom_param = theano.shared(value=val,borrow=True)
                delta = mom_param * self.momentum - gparam * self.lr # update the momentum of gradient (similar to running avg.)
                if self.nesterov:
                    new_delta = delta * self.momentum - gparam * self.lr
                else:
                    new_delta = delta
                updates.append((param, param + new_delta))
                updates.append((mom_param, delta))
            else:
                delta = -gparam * self.lr
                updates.append((param, param + delta))
        
        return updates