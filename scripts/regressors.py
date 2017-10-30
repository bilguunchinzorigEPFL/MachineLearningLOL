import numpy as np
from scripts.supportFunctions import *
import scripts.proj1_helpers as helper
from Probit import *
#These are the main regression functions which returns weights and losses
#1 least square gd
def least_squares_GD(y, tx, max_iters, gamma, initial_w=None):
    #initializing the weights
    if initial_w==None:
        initial_w = weight_init(tx.shape[1])
    ws = [initial_w]
    w = initial_w
    tr_idx,te_idx=helper.split_data(y,0.8)
    log=[]
    for n_iter in range(max_iters):
        g, loss=gradient_mse(y[tr_idx],tx[tr_idx],w)
        w=w-gamma*g
        line=helper.log_line(y[te_idx],np.dot(tx[te_idx],w),[n_iter,loss])
        log.append(line)
        print('Iter={i},Err={err},MSE={loss}'.format(i=line[1],err=line[0],loss=loss))
    return w, log

#2 stochastic gradient descent
def least_squares_SGD(y, tx, batch_size, max_iters, gamma, initial_w=None):
    if initial_w==None:
        initial_w = weight_init(tx.shape[1])
    ws = [initial_w]
    tr_idx,te_idx=helper.split_data(y,0.8)
    log=[]
    w = initial_w
    for n_iter in range(max_iters):
        g, loss=gradient_mse_sto(y[tr_idx],tx[tr_idx],w,batch_size)
        w=w-gamma*g
        # store w and loss
        ws.append(w)
        line=helper.log_line(y[te_idx],np.dot(tx[te_idx],w),[n_iter,loss])
        log.append(line)
        print(line)
    return w, log

#3 least squaress
def least_squares(y, tx):
    tr_idx,te_idx=helper.split_data(y,0.8)
    w=np.dot(np.linalg.inv(np.dot(np.transpose(tx[tr_idx]),tx[tr_idx])),np.dot(np.transpose(tx[tr_idx]),y[tr_idx]))
    mse=loss_mse(y[te_idx], tx[te_idx], w)
    # log=helper.log_line(y[te_idx],np.dot(tx[te_idx],w),[mse])
    # print(log)
    return w, mse

#4 ridge regression
def ridge_regression(y, tx, lambda_): 
    tr_idx,te_idx=helper.split_data(y,0.8)   
    w=np.dot(np.linalg.inv(np.dot(np.transpose(tx[tr_idx]),tx[tr_idx])+lambda_*np.eye(tx.shape[1])),np.dot(np.transpose(tx[tr_idx]),y[tr_idx]))
    mse=loss_mse(y[te_idx], tx[te_idx], w)
    log=helper.log_line(y[te_idx],np.dot(tx[te_idx],w),[mse])
    print(log)
    return w, log

#5 Logistic Regression 
#5.1 logistic_regression: function that retrieves the weights using logistic regression
# Output: 
# w: vector of trained weights. (Dx1)   
# logistic_loss: loss eveluated with the trained weights, i.e. the negative of the log-likelihood 
 # Arguments:   
        # y: Dependent variable (N x 1)  
        # x: Design Matrix (N x D). With N= number of features. D= number of attributes
        # max_iters: maximum number of iterations
        # gamma: Size of step in Gradient Descent. Belongs to the closed interval (0,1) 
        # w_initial: Vector of initial weights. 

def logistic_regression(y,x,max_iters,gamma,w_initial):
    log_loss=[]    # Vector storage of the loss in each iteration
    w = w_initial  
    for n_iter in range(max_iters):
        logistic_loss=-logistic_log_likelihood(y,x,w)  # Calculation of the log_likelihood 
        grad=logistic_gradient(y,x,w)                  # gradient 
        w=w-gamma*grad                                 # updating the weight 
        log_loss.append([log_loss])                    # loss storage 
        #raise NotImplementedError
        print("Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=logistic_loss))
    return  w, logistic_loss


#5.2 logistic_regression_regularized: function that retrieves the weights using logistic regression with l-2 regularization term
    # Output: 
        # w: vector of trained weights. (Dx1)   
        # logistic_loss: loss eveluated with the trained weights, i.e. the negative of the log-likelihood 
    # Arguments:   
        # y: Dependent variable (N x 1)  
        # x: Design Matrix (N x D). With N= number of features. D= number of attributes
        # max_iters: maximum number of iterations
        # gamma: Size of step in Gradient Descent. Belongs to the closed interval (0,1) 
        # w_initial: Vector of initial weights.
        # lambda_: penalizing term. Belongs to the postive real numbers. 

def logistic_regression_regularized(y,x,max_iters,gamma,w_initial,lambda_):
    log_loss=[] # Vector storage of the loss in each iteration
    w = w_initial
    for n_iter in range(max_iters):
        logistic_loss=-logistic_log_likelihood(y,x,w)    # Calculation of the log_likelihood i.e the negative of the loss function
        grad=logistic_gradient(y,x,w)               # Obtaining the losgistic gradient
        w=w-gamma*grad+lambda_*w                    # Updating the weight. We add the gradient of the penalizing term
        log_loss.append([logistic_like])            # loss storage 
        print("Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=logistic_loss))
    return  w, logistic_loss
    
#5.3 logistic_regression2: function that retrieves the weights using logistic regression.
# In each iteration we only allow the update of the weights if the loss reduced.
# In order to avoid to get stuck in local minima we add a gaussian random noise vector. 
    # Output: 
        # w: vector of trained weights. (Dx1)   
        # logistic_loss: loss eveluated with the trained weights, i.e. the negative of the log-likelihood 
    # Arguments:   
        # y: Dependent variable (N x 1)  
        # x: Design Matrix (N x D). With N= number of features. D= number of attributes
        # max_iters: maximum number of iterations
        # gamma: Size of step in Gradient Descent. Belongs to the closed interval (0,1) 
        # w_initial: Vector of initial weights. 

def logistic_regression2(y,x,max_iters,gamma,w_initial):
    logistic_loss=[] # Vector for storing logistic loss
    w = w_initial
    for n_iter in range(max_iters):
        # Calculation of the loss with weight of iteration n-1 (i.e. negative of the logistic likelihood)
        log_loss=-logistic_log_likelihood(y,x,w)      
        # Gradient    
        grad=logistic_gradient(y,x,w)                      
        # Update of the weight in iteration n
        w_1=w-gamma*grad+np.random.normal(0,100,x.shape[1])
        # Calculation of the loss with weight of iteration n (i.e. negative of the logistic likelihood)
        log_loss_1=-logistic_log_likelihood(y,x,w_1)
        # Condition: Change the weight in interation n if the loss is lower than in iteration n-1 
        if log_loss_1<log_loss:
            w=w_1
        else :
            w=w
        logistic_loss.append([log_loss])
        #raise NotImplementedError
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
             bi=n_iter, ti=max_iters - 1,l=log_loss))
    return  w, logistic_loss

#5.4 logistic_regression_regularized2: function that retrieves the weights using logistic regression with l-2 regularizer.
# In each iteration we only allow the update of the weights if the loss reduced.
# In order to avoid to get stuck in local minima we add a gaussian random noise vector. 
    # Output: 
        # w: vector of trained weights. (Dx1)   
        # logistic_loss: loss eveluated with the trained weights, i.e. the negative of the log-likelihood 
    # Arguments:   
        # y: Dependent variable (N x 1)  
        # x: Design Matrix (N x D). With N= number of features. D= number of attributes
        # max_iters: maximum number of iterations
        # gamma: Size of step in Gradient Descent. Belongs to the closed interval (0,1) 
        # w_initial: Vector of initial weights. 
        # lamba_: Penalizin term. It belongs to the postive reals. 

def logistic_regression_regularized2(y,x,max_iters,gamma,w_initial,lambda_):
    log_loss=[]  # initilizing vector for storage of losses
    w = w_initial   
    for n_iter in range(max_iters):
        # Calculation logistic loss of w obtained in iteration n-1 (i.e. the negative of the logistic likelihood)
        logistic_loss=-logistic_log_likelihood(y,x,w) 
        # Gradient 
        grad=logistic_gradient(y,x,w)
        # Update of the weight addint the gradient of the l-2 regularizer and a vector
        # of standard normal variables to avoid staying in local minima.
        w_1=w-gamma*grad+lambda_*w+np.random.normal(0,100,x.shape[1])
        # Calculation of the loss of w obtained in ietartion n 
        log_loss_1=logistic_log_likelihood(y,x,w_1)
        # Condition to change the weight only if the loss in iteration n is lower than the one of iteration n-1
        if log_loss_1<log_loss:
            w=w_1
        else :
            w=w
        log_loss.append([logistic_loss])
        print("Gradient Descent({bi}/{ti}): like={l}".format(
             bi=n_iter, ti=max_iters - 1,l=logistic_loss))
    return  w, logistic_loss    

#5.5 stoch_log_regr: function that retrieves the weights using logistic regression with stochastic gradient descent 
# Output: 
    # w: vector of trained weights. (Dx1)   
    # logistic_loss: loss eveluated with the trained weights, i.e. the negative of the log-likelihood 
 # Arguments:   
        # y: Dependent variable (N x 1)  
        # x: Design Matrix (N x D). With N= number of features. D= number of attributes
        # w_initial: Vector of initial weights. 
        # batch_size: size of the batch. Belongs to the interval (0,N)
        # max_iters: maximum number of iterations
        # gamma: Size of step in Gradient Descent. Belongs to the closed interval (0,1) 
        

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
    log=[]
    for i in range(max_iter):
        # get loss and update w.
        g=gradient_logistic(y[tr_idx],tx[tr_idx],w)
        w=w-gamma*g
        loss=loss_logistic(y[te_idx],tx[te_idx],w)
        # log info
        line=helper.log_line(y[te_idx],sigmoid(np.dot(tx[te_idx],w)),[i,loss],0.347)
        log.append(line)
        print(line)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, log

#Bilguun Logistic------------------------------------------------------------------------------
def logistic_regression_bilguun_reg(y, tx, threshold=1e-8, gamma=0.1,lambda_=0.1):
    # init parameters
    max_iter = 10000
    losses = []
    y=np.matrix(y).transpose()
    w = np.matrix(weight_init(tx.shape[1])).transpose()
    tr_idx,te_idx=helper.split_data(y,0.8)
    # start the logistic regression
    log=[]
    for i in range(max_iter):
        # get loss and update w.
        g=gradient_logistic(y[tr_idx],tx[tr_idx],w)
        reg_grad,reg_loss=regulizer(w,lambda_)
        w=w-gamma*(g+lambda_+reg_grad)
        loss=loss_logistic(y[te_idx],tx[te_idx],w)
        # log info
        line=helper.log_line(y[te_idx],sigmoid(np.dot(tx[te_idx],w)),[i,loss],0.347)
        log.append(line)
        print(line)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, log

#General regularized function
#TODO define functions here
def h(y):
    return 1
def ro(x):
    return x
def psi(y):
    return y
def A(x,exp_x):
    return np.log(1+exp_x)
def grad_ro(x): #gradient of ro without x
    return 1
def grad_A(x,exp_x):
    return exp_x/(1+exp_x)
#main calculation
def gen_likelihood(y,x,exp_x): #used h, Ro, Psi, A
    return np.sum(np.log(h(y))+ro(x)*psi(y)-A(x,exp_x))/y.shape[0]
def gen_gradient(y,tx,x,exp_x): #used Gradient Ro, Gradient A, Psi
    return np.dot(np.transpose(tx),(grad_ro(x)*psi(y)-grad_A(x,exp_x)))/y.shape[0]
#tester
def gen_pdf(y,x,exp_x):
    return (exp_x*y)/(1+exp_x)
def gen_tester(y,x,exp_x):
    y_pred=gen_pdf(1,x,exp_x)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    e=y-y_pred
    return np.mean(np.absolute(e))
#main regressor
def gen_grad_regression(y,tx,max_iter=1000,gamma=0.1,threshold=1e-8):
    w=np.zeros(tx.shape[1])
    tr_idx,te_idx=helper.split_data(y,0.9)
    losses=[]
    for i in range(0,max_iter):
        x=np.dot(tx,w)
        exp_x=np.exp(x)
        g=gen_gradient(y[tr_idx],tx[tr_idx],x[tr_idx],exp_x[tr_idx])
        like=gen_likelihood(y[te_idx],x[te_idx],exp_x[te_idx])
        loss=gen_tester(y[te_idx],x[te_idx],exp_x[te_idx])
        w=w+gamma*g
        if i%100==0:
            print("Current iteration={i}, loss={l}, like={like}".format(i=i, l=loss,like=like))
        losses.append(like)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w

#Cascader
def pred_grad_booster(tx,ws,alpha=1):
    y_pred=None
    first=True
    for w in ws:
        x=np.dot(tx,w)
        exp_x=np.exp(x)
        pred=gen_pdf(1,x,exp_x)
        if first:
            y_pred=alpha*pred
            first=False
        else:
            y_pred+=(alpha*pred-alpha)*(1+alpha)+1
    # y_pred[np.where(y_pred > 0.35)] = 1
    # y_pred[np.where(y_pred <= 0.35)] = 0
    return y_pred
def grad_booster(y,tx,num=20,alpha=1,threshold=1e-4):
    ws=[]
    orig_y=y
    log=[]
    losses=[]
    for n in range(0,num):
        w=gen_grad_regression(y,tx)
        ws.append(w)
        x=np.dot(tx,w)
        exp_x=np.exp(x)
        pred=gen_pdf(1,x,exp_x)
        y=(y-alpha*pred+alpha)/(1+alpha)
        pred=pred_grad_booster(tx,ws)
        err=np.mean(np.absolute(orig_y-pred))
        print("iter={i}, residual={r}, err={e}".format(i=n,r=np.mean(np.absolute(y)),e=err))
        log.append([n,np.mean(np.absolute(y)),err])
    return ws,log

# glr: Function to obtain Generalized Linear Regression 
# Output: 
#   w: Vector of estimated weights (D x 1). 
# Arguments:   
    # y: Dependent variable (N x 1)  
    # x: Design Matrix (N x D). With N= number of features. D= number of attributes
    # max_iters: maximum number of iterations
    # gamma: Size of step in Gradient Descent. Belongs to the closed interval (0,1) 
    # w_initial: Vector of initial weights. Can choos three options:
    #   1. zeros: 
    #   2. ones
    #   3. random: random initialization from a Uniform distribution with parameters (0,1)
    # type: Type of Generalized Linear Regression. Options Available:
    #       1. logistic
    #       2. probit 

def glr(y,x,max_iters=1,gamma=0.5,w_initial='zeros',type='logistic'):
    loss=[]
    if w_initial=='zeros': 
        w_initial=np.zeros((x.shape[1]))
    elif w_initial=='ones':    
        w_initial=np.ones((x.shape[1]))
    else: 
        w_initial=np.random.rand(x.shape[1])
    w = w_initial
    if type=='logistic':
        for n_iter in range(max_iters):
            loss_log=-logistic_log_likelihood(y,x,w)
            grad=logistic_gradient(y,x,w)
            w_1=w-gamma*grad+np.random.normal(0,100,x.shape[1])
            loss_log1=-logistic_log_likelihood(y,x,w_1)
            if loss_log1<loss_log:
                w=w_1
            else :
                w=w
            loss.append([loss_log])
            #raise NotImplementedError
            print("Logistic Regression{bi}/{ti}): loss={l}".format(
             bi=n_iter, ti=max_iters - 1,l=loss))
    else: 
        for n_iter in range(max_iters):
            loss_prob=-probit_log_likelihood(y,x,w)
            grad=probit_gradient(y,x,w)
            w_1=w-gamma*grad+np.random.normal(0,100,x.shape[1])
            loss_prob1=-probit_log_likelihood(y,x,w_1)
            if loss_prob1<loss_prob:
                w=w_1
            else :
                w=w
            loss.append([loss_prob])
            print("Probit Regression{bi}/{ti}): loss={l}".format(
             bi=n_iter, ti=max_iters - 1,l=loss))
    return  w
          
