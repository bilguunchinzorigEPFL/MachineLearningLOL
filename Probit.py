from scipy.stats import norm
import numpy as np 

# probit_cdf: Function that gives the probability that y_i=1, i.e. y_i=b. 
# Output
#   probit_cdf: vector of cdf of x_i'w which is equivalent to P(y_i=1)    
# Inputs 
#   y= dependent variable 
#   x= design matrix
#   w= weights 
def probit_cdf(y,x,w):
    z=np.dot(x,w)
    probit_cdf=norm.cdf(z,loc=0,scale=1)
    return probit_cdf

# probit_pdf: Function that gives the probability density function of x_i'w
# Output
#   probit_pdf: vector of cdf of x_i'w which is equivalent to P(y_i=1)    
# Inputs 
#   y= dependent variable 
#   x= design matrix
#   w= weights 
def probit_pdf(y,x,w):
    z=np.dot(x,w)
    probit_pdf=norm.pdf(z,loc=0,scale=1)
    return probit_pdf

# probit_log_likelihood: Function that gives the log-likelihood of x_i'w
# Output
#   probit_log_likelihood: vector of cdf of x_i'w which is equivalent to P(y_i=1)    
# Inputs 
#   y= dependent variable 
#   x= design matrix
#   w= weights 
def probit_log_likelihood(y,x,w):
    p_cdf=probit_cdf(y,x,w)
    probit_log_likelihood=np.dot(np.transpose(y),p_cdf)+np.dot(np.transpose(y-np.ones((len(y)))),np.ones((len(y)))-p_cdf)
    return probit_log_likelihood


# probit_gradient: Function that gives the gradient of the probit regression
# Output
#   log_grad: gradient of the probit regression. 
# Inputs 
#   y= dependent variable 
#   x= design matrix
#   w= weights 	
def probit_gradient(y,x,w):
    pdf=probit_pdf(y,x,w)
    cdf=probit_cdf(y,x,w)
    probit_grad=np.dot((np.dot(np.transpose(y),pdf/cdf)+np.dot(np.transpose((np.ones((len(y)))-y)),pdf/(np.ones((len(y)))-cdf))),x)
    #probit_grad=np.dot((np.dot(np.transpose(y),np.divide(pdf,cdf)+np.dot(np.transpose((np.ones((len(y)))-y)),np.divide(pdf,(np.ones((len(y)))-cdf)))),x))
    return probit_grad

