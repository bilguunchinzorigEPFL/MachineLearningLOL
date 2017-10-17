import numpy as np
import supportFunctions *

from scipy.stats import logistic

def logistic_gradient(y,x,w):
    log_grad=np.dot(np.transpose(tx),(y-logistic.pdf(np.dot(tx,w), loc=0, scale=1)))
    return log_grad
	
def logistic_regression(y,x,max_iters,gamma,w_initial):
    w=[w_initial]
    losses = []
    w = w_initial
    for n_iter in range(max_iters):
        loss=loss_mse(y,tx,w)
        w=w+gamma*logistic_gradient(y,tx,w)
        losses.append([loss])
        #raise NotImplementedError
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
             bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, w
    
    
def compute_stoch_logistic_gradient(y, tx, w, batch_size):
    index=np.int_(np.floor(len(y)*np.random.uniform(0,1,batch_size)))
    y=y[np.transpose(index)]
    tx=tx[np.transpose(index),:] 
    stoch_log_gradient=logistic_gradient(y, tx, w)
    raise NotImplementedError
    return stoch_log_gradient
    
    
def stoch_log_regr(
    y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        index=np.int_(np.floor(len(y)*np.random.uniform(0,1,batch_size)))
        y=y[np.transpose(index)]
        tx=tx[np.transpose(index),:] 
        loss=loss_mse(y,tx,w)
        w=w+gamma*logistic_gradient(y,x,w)
        losses.append([loss])
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
