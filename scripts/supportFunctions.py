import numpy as np
#LOSS calculators
def loss_mse(y,tx,w):
    e=y-np.dot(tx,w) #TODO: figure this out
    return sum(e**2)/(y.shape[0])
def loss_rmse(y,tx,w):
    return np.sqrt(loss_mse(y,tx,w))

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
#for example if x.shape=[n,f] then it returns [n,f*degree] sized matrix
def build_poly(x, degree):
    degree+=1
    powers=np.ones([x.shape[1]*degree,x.shape[0]])
    for f in range(0,x.shape[1]):
        for d in range(0,degree):
            powers[:,f*degree+d]=x[:,f]**d
    return powers

#Weight initializer
def weight_init(size,lower=0,upper=1):
#<<<<<<< HEAD
    return np.random.rand(size)*(upper-lower)+lower

#Logistic regression

def logistic_pdf(y,tx,w):
	logistic_pdf=np.ones((len(y)))/(np.ones((len(y)))+np.exp(-np.dot(tx,w)))
	return logistic_pdf
	
def logistic_gradient(y,tx,w):
    log_grad=np.dot(np.transpose(tx),(y-logistic_pdf(y,tx,w)))
    return log_grad

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
    
