import numpy as np


# LOSS calculators
def sigmoid(t):
    return 1/(1+np.exp(-t))     
def loss_logistic(y, tx, w):
    x=np.dot(tx,w)
    return np.mean(np.log(1+np.exp(x))-np.multiply(x,y))
def loss_mse(y,tx,w):
    e=y-np.dot(tx,w) #TODO: figure this out
    return np.mean(e**2)
def loss_rmse(y,tx,w):
    return np.sqrt(loss_mse(y,tx,w))
def loss_abs(y,tx,w):
    e=y-np.around(np.dot(tx,w))
    return np.mean(np.absolute(e),axis=0)
#GRADIENT and loss calculators

#mean square error gradient
def gradient_mse(y,tx,w):
    tmp=np.dot(tx,w)-y
    return np.dot(tx.transpose(),tmp)/(y.shape[0]), np.mean((tmp**2))

#root mean square error gradient
def gradient_rmse(y,tx,w):
    grad, loss=gradient_mse(y,tx,w)
    return grad/loss, np.sqrt(loss)

#stochastic mean square error gradient
def gradient_mse_sto(y,tx,w,batch_size):
    #getting random sample
    target=np.random.choice(range(0,len(y)),size=batch_size)
    #calculating the gradient
    return gradient_mse(y[target],tx[target],w)
#regularization parameter gradient and loss
def regulizer(w,lambda_):
    reg=np.absolute(w)
    return lambda_*reg,lambda_*np.sum(np.multiply(reg,reg))

#function specific to polynomial regression
#creating data for polynomial regression which returns all the degree of x value
#for example if x.shape=[n,f] then it returns [n,f* ] sized matrix
def build_poly(x, degree):
    powers=np.ones([x.shape[0],x.shape[1]*degree])
    for d in range(0,degree):
        powers[:,(d*x.shape[1]):((d+1)*x.shape[1])]=(x**(d+1))/1000
    return powers
# Function to obtain radial basis 
#def build_radialbasis(x, type=gaussian): 
    # Arguments: X= Matrix of features 
    # Type: type of radial basis. Options available: 1. Gaussian, 2. Thin plate spline 
 #   if type=Gaussian:


  #  else:
        
   # powers=np.ones([x.shape[0],x.shape[1]*degree])
   # for f in range(0,x.shape[1]):
    #    for d in range(1,degree):
     #       powers[:,f*degree+d]=x[:,f]**d
    #return powers
# Function to obtain polar basis 

def gradient_logistic(y, tx, w):
    x=np.dot(tx,w)
    tmp=sigmoid(x)-y
    return np.dot(tx.transpose(),tmp)/y.shape[0]

# Build polynomial for chosen features in the raw vector pol

def build_poly_reg(x,pol, degree):
    power=x
    for f in range(0,len(pol)):
    	for d in range(2,degree):
	        power=np.c_[power,x[:,pol[f]]**d]
    return power

# Obtain radial basis 

#Weight initializer
def weight_init(size,lower=0,upper=1):
    return np.random.rand(size)*(upper-lower)+lower

# logistic_cdf: Function that gives the probability that y_i=1, i.e. y_i=b. 
# Output
#   logistic_cdf: vector of cdf of x_i'w which is equivalent to P(y_i=1)    
# Inputs 
#   y= dependent variable 
#   x= design matrix
#   w= weights 
def logistic_cdf(y,x,w):
    logistic_cdf=np.exp(np.dot(x,w))/(np.ones((len(y)))+np.exp(np.dot(x,w)))
    return logistic_cdf

# logistic_gradient: Function that gives the gradient of the logistic regression
# Output
#   log_grad: gradient of the logistic regression. 
# Inputs 
#   y= dependent variable 
#   x= design matrix
#   w= weights 	
def logistic_gradient(y,x,w):
    log_grad=np.dot(np.transpose(x),(logistic_cdf(y,x,w)-y))/(len(y))
    return log_grad

# stoch_logistic_gradient: Function that gives the stochastic gradient of the logistic regression
# Output
#   stoch_log_gradient: stochastic gradient of the logistic regression. 
# Arguments
#   y= dependent variable 
#   x= design matrix
#   w= weights 	
#   batch size= size of the batch. It belongs to the closed interval (0,N)
def stoch_logistic_gradient(y, x, w, batch_size):
    index=np.int_(np.floor(len(y)*np.random.uniform(0,1,batch_size)))
    y=y[np.transpose(index)]
    x=x[np.transpose(index),:] 
    stoch_log_gradient=logistic_gradient(y, x, w)
    return stoch_log_gradient
	
# logistic_log_likelihood: Function that gives the the logistic log likelihood, i.e the negative of the loss of the logistic regression
# Output
#   logistic_log_likelihood:  the log-likelihood
# Arguments
#   y= dependent variable 
#   x= design matrix
#   w= weights 	
def logistic_log_likelihood(y,x,w):
	logisticdf=logistic_cdf(y,x,w)
	log_likelihood=np.dot(np.transpose(y),logisticdf)+np.dot(np.transpose(np.ones((len(y)))-y),np.ones((len(y)))-logisticdf)
	return log_likelihood

          
