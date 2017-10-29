from scipy.stats import norm

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
    probit_grad=np.dot((np.dot(np.transpose(y),probit_pdf/probit_cdf)+np.dot(np.transpose((1-y)),probit_pdf((1-probit_cdf))),x)
    return probit_grad
