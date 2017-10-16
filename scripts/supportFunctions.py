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
#regularization parameter gradient
def gradient_reg(w,lambda_):
    return 2*lamb*w

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
    return np.random.rand(size)*(upper-lower)+lower