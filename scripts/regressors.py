from scripts.supportFunctions import *
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
    mse=loss_mse(y, tx, w)
    return w, mse

#4 ridge regression    
def ridge_regression(y, tx, lambda_):    
    w_ridge=np.dot(np.linalg.inv(np.dot(np.transpose(tx),tx)+lambda_*np.eye(tx.shape[1])),np.dot(np.transpose(tx),y))
    mse_ridge=loss_mse(y, tx, w_ridge)
    return w_ridge, mse_ridge

#5 Logistic Regression 
#5.1 GD
def logistic_regression(y,tx,max_iters,gamma,w_initial=None):
    if w_initial==None:
        w_initial = weight_init(tx.shape[1])
    w=[w_initial]
    losses = []
    log_likelihood=[]
    w = w_initial
    for n_iter in range(max_iters):
        log_like=logistic_log_likelihood(y,tx,w)
        grad=logistic_gradient(y,tx,w)
        w=w+gamma*grad
        log_likelihood.append([log_like])
        print("Logistic Regression Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=log_like))
    return  w, losses

#5.1  Stochastic GD 
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