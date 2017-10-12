#1 least square
def least_squares(y, tx):
    txt=tx.transpose()
    #just impementation of the equation (XtX)^-1 * Xt * y = w
    w=np.matmul(np.linalg.inv(np.matmul(txt,tx)),np.matmul(txt,y))
    return 0, w

#2 stochastic gradient descent
def compute_stoch_gradient(y, tx, w):
    #getting random sample
    target=np.random.randint(y.shape[0])
    #calculating the gradient
    tmp=sum(np.multiply(tx[target,:],w))
    return tmp*tx[target,:]

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

#5,6 polynomial
#preparing data to polynomial
def power_iterator(powers,degree,current):
    if(end):
        return powers[f,d,:]
    tmp=power_iterator()
    result=empty array
    for d in range(0,degree):
        result concat(np.multiply(powers[f,d,:]*tmp))
    return result

def build_poly(x, degree):
    degree+=1
    powaa=np.ones([x.shape[0],degree**x.shape[1]])
    
    return powaa
#calculating gradient based on RMSE loss function
def poly_grad(y,tx,w,lamb):
    rows,cols=np.indices(tx.shape)
    tmp=y-np.sum(np.multiply(w[cols],tx),axis=1)
    loss=np.sqrt(np.sum((tmp)**2)/(y.shape[0]))
    tmp=np.sum(np.multiply(tmp[rows],tx),axis=0)/(-y.shape[0])
    #ridge regression
    reg=2*lamb*w
    return (tmp/loss)+reg, loss
    #return tmp/loss, loss
#just typical descend algorithm    
def poly_desc(y, tx, initial_w, max_iters, gamma,lamb):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g, loss=poly_grad(y,tx,w,lamb)
        w=w-gamma*g
        ws.append(w)
        losses.append(loss)
    return losses, w