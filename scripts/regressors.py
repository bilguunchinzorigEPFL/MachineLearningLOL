from supportFunctions import *
#These are the main regression functions which returns weights and losses
#1 least square gd
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g=compute_gradient(y,tx,w)
        loss=compute_loss(y,tx,w)
        w=w-gamma*g
        ws.append(w)
        losses.append(loss)
    return losses, ws

#2 stochastic gradient descent
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g=compute_stoch_gradient(y,tx,w)
        #since loss is square of gradient[0] where x0 is equal to 1
        loss=g[0]**2
        w=w-gamma*g
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return losses, ws

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

#5 polynomial  
def poly_descent(y, x, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    tx=build_poly(x,degree)
    for n_iter in range(max_iters):
        g, loss=gradient_rmse(y,x,w)
        w=w-gamma*g
        ws.append(w)
        losses.append(loss)
    return losses, w

#6 polynomial regularized
def poly_descent(y, tx, initial_w, max_iters, gamma,lambda_):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g, loss=gradient_rmse(y,x,w)
        w=w-gamma*(g+gradient_reg(w,lambda_))
        ws.append(w)
        losses.append(loss)
    return losses, w