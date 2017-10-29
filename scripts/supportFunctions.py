import numpy as np


# LOSS calculators
def sigmoid(t):
    return 1/(1+np.exp(-t))     
def loss_logistic(y, tx, w):
    x=np.dot(tx,w)
    e=np.around(sigmoid(x))-y
    return np.mean(np.absolute(e))
def loss_mse(y,tx,w):
    e=y-np.dot(tx,w) #TODO: figure this out
    return sum(e**2)/(y.shape[0])
def loss_rmse(y,tx,w):
    return np.sqrt(loss_mse(y,tx,w))
def loss_abs(y,tx,w):
    e=y-np.around(np.dot(tx,w))
    return np.mean(np.absolute(e),axis=0)
#GRADIENT and loss calculators

#mean square error gradient
def gradient_mse(y,tx,w):
    rows,cols=np.indices(tx.shape)
    tmp=y-np.sum(np.multiply(w[cols],tx),axis=1)
    return np.sum(np.multiply(tmp[rows],tx),axis=0)/(-y.shape[0]), sum(tmp**2)/(y.shape[0])

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
    return lambda_*reg,lambda_*np.sum(reg**2)

#function specific to polynomial regression
#creating data for polynomial regression which returns all the degree of x value
#for example if x.shape=[n,f] then it returns [n,f* ] sized matrix
def build_poly(x, degree):
    degree+=1
    powers=np.ones([x.shape[0],x.shape[1]*degree])
    for f in range(0,x.shape[1]):
        for d in range(1,degree):
            powers[:,f*degree+d]=x[:,f]**d
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

#Logistic regression

#def logistic_pdf(tx,w):
#	logistic_pdf=np.ones((tx.shape[0]))/(np.ones((tx.shape[0]))+np.exp(-(np.dot(tx,w))))
#	return logistic_pdf

def logistic_pdf(y,tx,w):
    logistic_pdf=np.exp(np.dot(tx,w))/(np.ones((len(y)))+np.exp(np.dot(tx,w)))
    return logistic_pdf
	
def logistic_gradient(y,tx,w):
    log_grad=np.dot(np.transpose(tx),(y-logistic_pdf(y,tx,w)))/(len(y))
    return log_grad
	
	
#def logistic_gradient(y,tx,w):
 #   log_grad=np.dot(np.transpose(tx),(y-logistic_pdf(tx,w)))
  #  return log_grad

#def logistic_log_likelihood(y,tx,w):
#	logisticpdf=logistic_pdf(tx,w)
#	log_likelihood=np.dot(np.transpose(y),logisticpdf)+np.dot(np.transpose(np.ones((len(y)))-y),np.ones((len(y)))-logisticpdf)
#	return log_likelihood

def logistic_log_likelihood(y,tx,w):
	logisticpdf=logistic_pdf(y,tx,w)
	log_likelihood=np.dot(np.transpose(y),logisticpdf)+np.dot(np.transpose(np.ones((len(y)))-y),np.ones((len(y)))-logisticpdf)
	return log_likelihood
          
def compute_stoch_logistic_gradient(y, x, w, batch_size):
    index=np.int_(np.floor(len(y)*np.random.uniform(0,1,batch_size)))
    y=y[np.transpose(index)]
    x=x[np.transpose(index),:] 
    stoch_log_gradient=logistic_gradient(y, x, w)
    return stoch_log_gradient
