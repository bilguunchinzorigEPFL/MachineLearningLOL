import numpy as np
from scripts.supportFunctions import *
import scripts.supportFunctions as sp
import scripts.proj1_helpers as helper
#These are the main regression functions which returns weights and losses
#1 least square gd
def gradient_descent(y, tx, max_iters, gamma, initial_w=None):
    #initializing the weights
    if initial_w==None:
        initial_w = weight_init(tx.shape[1])
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g, loss=gradient_mse(y,tx,w)
        w=w-gamma*g
        ws.append(w)
        losses.append(loss)
    return w, losses, ws

#2 stochastic gradient descent
def stochastic_gradient_descent(y, tx, batch_size, max_iters, gamma, initial_w=None):
    if initial_w==None:
        initial_w = weight_init(tx.shape[1])
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g, loss=gradient_mse_sto(y,tx,w,batch_size)
        w=w-gamma*g
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return w, losses, ws

#3 least squaress
def least_squares(y, tx):
    w=np.dot(np.linalg.inv(np.dot(np.transpose(tx),tx)),np.dot(np.transpose(tx),y))
    #mse=loss_mse(y, tx, w)
    return w

#4 ridge regression    
def ridge_regression(y, tx, lambda_):    
    w_ridge=np.dot(np.linalg.inv(np.dot(np.transpose(tx),tx)+lambda_*np.eye(tx.shape[1])),np.dot(np.transpose(tx),y))
    mse_ridge=loss_mse(y, tx, w_ridge)
    return w_ridge, mse_ridge

#5 Logistic Regression 
#5.1 GD

def logistic_regression(y,tx,max_iters,gamma,w_initial):
    log_likelihood=[]
    w = w_initial
    for n_iter in range(max_iters):
        log_like=logistic_log_likelihood(y,tx,w)
        grad=logistic_gradient(y,tx,w)
        w=w+gamma*grad
        log_likelihood.append([log_like])
        #raise NotImplementedError
        print("Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=log_like))
    return  w

def logistic_regression_regularized(y,tx,max_iters,gamma,w_initial,lambda_):
    log_likelihood=[]
    w = w_initial
    for n_iter in range(max_iters):
        log_like=logistic_log_likelihood(y,tx,w)
        grad=logistic_gradient(y,tx,w)
        w=w+gamma*grad-lambda_*w
        log_likelihood.append([log_like])
        #raise NotImplementedError
        print("Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=log_like))
    return  w
    
#5.1.1 GD
def logistic_regression2(y,tx,max_iters,gamma,w_initial):
    log_likelihood=[]
    w = w_initial
    for n_iter in range(max_iters):
        log_like=logistic_log_likelihood(y,tx,w)
        grad=logistic_gradient(y,tx,w)
        w_1=w+gamma*grad+np.random.normal(0,100,tx.shape[1])
        log_like_1=logistic_log_likelihood(y,tx,w_1)
        if log_like_1>log_like:
            w=w_1
        else :
            w=w
        log_likelihood.append([log_like])
        #raise NotImplementedError
        print("Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=log_like))
    return  w
    
def logistic_regression_regularized2(y,tx,max_iters,gamma,w_initial,lambda_):
    log_likelihood=[]
    w = w_initial
    for n_iter in range(max_iters):
        log_like=logistic_log_likelihood(y,tx,w)
        grad=logistic_gradient(y,tx,w)
        w_1=w+gamma*grad-2*lambda_*w+np.random.normal(0,100,tx.shape[1])
        log_like_1=logistic_log_likelihood(y,tx,w_1)
        if log_like_1>log_like:
            w=w_1
        else :
            w=w
        log_likelihood.append([log_like])
        #raise NotImplementedError
        print("Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=log_like))
    return  w    

#5.2  Stochastic GD 
def stoch_log_regr(
    y, x, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        index=np.int_(np.floor(len(y)*np.random.uniform(0,1,batch_size)))
        y=y[np.transpose(index)]
        x=x[np.transpose(index),:] 
        log_like=logistic_log_likelihood(y,tx,w)
        w=w+gamma*logistic_gradient(y,x,w)
        log_likelihood.append([log_like])
        print("Logistic Regression Stochastic Gradient Descent({bi}/{ti}):  like={l}".format(
              bi=n_iter, ti=max_iters - 1, l=log_like))
    return w, losses

#6 Logistic with regularization
def logistic_regression_reg(y,tx,max_iters,gamma,lambda_,w_initial=None):
    if w_initial==None:
        w_initial = weight_init(tx.shape[1])
    w=[w_initial]
    losses = []
    log_likelihood=[]
    w = w_initial
    for n_iter in range(max_iters):
        log_like=logistic_log_likelihood(y,tx,w)
        grad=logistic_gradient(y,tx,w)
        reg_grad,reg_loss=regulizer(w,lambda_)
        w=w+gamma*(grad+reg_grad)
        log_likelihood.append([log_like+reg_loss])
        print("Logistic Regression Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=log_like))
    return  w, losses


#Bilguun Logistic------------------------------------------------------------------------------
def logistic_regression_bilguun(y, tx, threshold=1e-8, gamma=0.1):
    # init parameters
    max_iter = 10000
    losses = []
    y=np.matrix(y).transpose()
    w = np.matrix(weight_init(tx.shape[1])).transpose()
    tr_idx,te_idx=helper.split_data(y,0.8)
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        g=gradient_logistic(y[tr_idx],tx[tr_idx],w)
        w=w-gamma*g
        loss=loss_logistic(y[te_idx],tx[te_idx],w)
        # log info
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

#Bilguun Logistic------------------------------------------------------------------------------
def logistic_regression_bilguun_reg(y, tx, threshold=1e-8, gamma=0.1,lambda_=0):
    # init parameters
    max_iter = 10000
    losses = []
    y=np.matrix(y).transpose()
    w = np.matrix(weight_init(tx.shape[1])).transpose()
    tr_idx,te_idx=helper.split_data(y,0.8)
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        g=gradient_logistic(y[tr_idx],tx[tr_idx],w)
        reg_grad,reg_loss=regulizer(w,lambda_)
        w=w-gamma*(g+lambda_+reg_grad)
        loss=loss_logistic(y[te_idx],tx[te_idx],w)
        # log info
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

#General regularized function
#TODO define functions here
def h(y):
    return 1
def ro(x,var):
    return np.array([x/var,[-0.5/var]*x.shape[0]]).transpose()
def psi(y):
    return np.array([y,y**2]).transpose()
def A(x,var):
    return (x**2)/(2*var)
def grad_ro(x,var):#gradient of ro without x
    return np.array([[1/var]*x.shape[0],[0]*x.shape[0]]).transpose()
def grad_A(x,var):
    return x/var
#main calculation
def gen_likelihood(y,x,var): #used h, Ro, Psi, A
    return np.sum(np.log(h(y))+np.sum(ro(x,var)*psi(y),axis=1)-A(x,var))/y.shape[0]
def gen_gradient(y,tx,x,var): #used Gradient Ro, Gradient A, Psi
    return np.dot(np.transpose(tx),(np.sum(grad_ro(x,var)*psi(y),axis=1)-grad_A(x,var)))/y.shape[0]
#tester
def gen_pdf(y,x,var):
    return np.exp(-((y-x)**2)/(2*var))
def gen_tester(y,x,var):
    y_pred=gen_pdf(1,x,var)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    e=y-y_pred
    return np.mean(np.absolute(e))
#main regressor
def gen_grad_regression(y,tx,var,max_iter=1000,gamma=0.1,threshold=1e-8):
    w=weight_init(tx.shape[1])
    tr_idx,te_idx=helper.split_data(y,0.9)
    losses=[]
    for i in range(0,max_iter):
        x=np.dot(tx,w)
        #exp_x=np.exp(x)
        g=gen_gradient(y[tr_idx],tx[tr_idx],x[tr_idx],var=var)
        like=gen_likelihood(y[te_idx],x[te_idx],var=var)
        loss=gen_tester(y[te_idx],x[te_idx],var=var)
        w=w+gamma*g
        if i%100==0:
            print("Current iteration={i}, loss={l}, like={like}".format(i=i, l=loss,like=like))
        losses.append(like)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w
def grad_sigma(y,x,var):
    e=((y-x)**2)/2
    return np.mean(e/(var**2))
#parameter estimation
def normal_regression(y,tx,max_iter=50,gamma=0.1,threshold=1e-8):
    var=weight_init(1)[0]
    tr_idx,te_idx=helper.split_data(y,0.9)
    losses=[]
    for i in range(0,max_iter):
        w=gen_grad_regression(y[tr_idx],tx[tr_idx],var=var,max_iter=101,gamma=0.1)
        x=np.dot(tx,w)
        var_grad=grad_sigma(y,x,var)
        loss=gen_tester(y[te_idx],x[te_idx],var=var)
        print("V_iter={i}, loss={l}, var={var}".format(i=i, l=(-var_grad*var),var=var))
        var=var+gamma*var_grad
    return var

#Cascader
def pred_grad_booster(tx,ws,alpha=1):
    y_pred=np.zeros(tx.shape[0])
    for w in ws:
        x=np.dot(tx,w)
        exp_x=np.exp(x)
        pred=gen_pdf(1,x,exp_x)
        y_pred=y_pred+alpha*pred-0.5
    y_pred[np.where(y_pred > 0)] = 1
    y_pred[np.where(y_pred <= 0)] = -1
    return y_pred
def grad_booster(y,tx,num=10,alpha=1):
    ws=[]
    orig_y=y
    for n in range(0,num):
        w=gen_grad_regression(y,tx)
        ws.append(w)
        x=np.dot(tx,w)
        exp_x=np.exp(x)
        pred=gen_pdf(1,x,exp_x)
        y=y-alpha*pred+0.5
        pred=pred_grad_booster(tx,ws)
        err=np.mean(np.absolute(orig_y-pred))
        print("iter={i}, residual={r}, err={e}".format(i=n,r=np.mean(np.absolute(y)),e=err))
    return ws