import scripts.regressors as reg
import scripts.proj1_helpers as helper
import scripts.supportFunctions as sp
import numpy as np
#load the data
y,tx,ids=helper.load_csv_data("data/train.csv",standard=True)

tx60=np.zeros((tx.shape[0],tx.shape[1]*2))
tx60[:,:tx.shape[1]]=tx
tx60[:,tx.shape[1]:]=tx**2
w=reg.logistic_regression_gradient_descent_demo(y, tx60)
print("Started")
def forward_selection(y,tx,threshold=1e-4):
    used=[]
    losses=[]
    left=set(range(0,tx.shape[1]))
    for i in range(0,tx.shape[1]):
        best=(999,0)
        left=left-set(used)
        for j in left:
            tmp_use=used[:]
            tmp_use.append(j)
            w,loss=reg.logistic_regression_gradient_descent_demo(y,tx[:,tmp_use])
            if(best[0]>loss):
                best=(loss,tmp_use)
            #print("j={j}, loss={loss}, tmp_use={tmp_use}".format(j=j,loss=loss,tmp_use=tmp_use))
        used=best[1]
        losses.append(best[0])
        print("Selected number of F={f}, loss={loss}".format(f=i,loss=best))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return used
def polynomial_maker(tx,degrees):
    step=tx.shape[1]
    result=np.zeros((tx.shape[0],step*len(degrees)))
    for i in range(0,len(degrees)):
        result[:,range(step*i,step*(i+1))]=tx**(degrees[i])
    return result
# print(forward_selection(y,degree(tx,3)))

#TODO define functions here
def h(y):
    return 1
def ro(x,var):
    return np.array([x/var,[-0.5/var]*x.shape[0]]).transpose()
def psi(y):
    return np.array([y,y**2]).transpose()
def A(x,var):
    return (x**2)/(2*var)
def grad_ro(x,var):#gradient of ro without x
    return np.array([[1/var]*x.shape[0],[0]*x.shape[0]]).transpose()
def grad_A(x,var):
    return x/var
#main calculation
def gen_likelihood(y,x,var): #used h, Ro, Psi, A
    return np.sum(np.log(h(y))+np.sum(ro(x,var)*psi(y),axis=1)-A(x,var))/y.shape[0]
def gen_gradient(y,tx,x,var): #used Gradient Ro, Gradient A, Psi
    return np.dot(np.transpose(tx),(np.sum(grad_ro(x,var)*psi(y),axis=1)-grad_A(x,var)))/y.shape[0]
#tester
def gen_pdf(y,x,var):
    return np.exp(-((y-x)**2)/(2*var))
def gen_tester(y,x,var):
    y_pred=gen_pdf(1,x,var)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    e=y-y_pred
    return np.mean(np.absolute(e))
#main regressor
def gen_grad_regression(y,tx,var,max_iter=1000,gamma=0.1,threshold=1e-8):
    w=weight_init(tx.shape[1])
    tr_idx,te_idx=helper.split_data(y,0.9)
    losses=[]
    for i in range(0,max_iter):
        x=np.dot(tx,w)
        #exp_x=np.exp(x)
        g=gen_gradient(y[tr_idx],tx[tr_idx],x[tr_idx],var=var)
        like=gen_likelihood(y[te_idx],x[te_idx],var=var)
        loss=gen_tester(y[te_idx],x[te_idx],var=var)
        w=w+gamma*g
        if i%100==0:
            print("Current iteration={i}, loss={l}, like={like}".format(i=i, l=loss,like=like))
        losses.append(like)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w
def grad_sigma(y,x,var):
    e=((y-x)**2)/2
    return np.mean(e/(var**2))
#parameter estimation
def normal_regression(y,tx,max_iter=50,gamma=0.1,threshold=1e-8):
    var=weight_init(1)[0]
    tr_idx,te_idx=helper.split_data(y,0.9)
    losses=[]
    for i in range(0,max_iter):
        w=gen_grad_regression(y[tr_idx],tx[tr_idx],var=var,max_iter=101,gamma=0.1)
        x=np.dot(tx,w)
        var_grad=grad_sigma(y,x,var)
        loss=gen_tester(y[te_idx],x[te_idx],var=var)
        print("V_iter={i}, loss={l}, var={var}".format(i=i, l=(-var_grad*var),var=var))
        var=var+gamma*var_grad
    return var


# #Parameter testing
# params=[]
# min_tuple=(999,0)
# te_loss_log=[]
# tr_loss_log=[]
# param=10
# while param!=0:
#     param=float(input("Please enter parameter value: "))
#     tr_idx,te_idx=helper.split_data(y,0.8)
#     w,tr_losses=reg.logistic_regression(y[tr_idx],tx[tr_idx],int(param),50) #TODO enter regressor
#     te_loss=sp.logistic_log_likelihood(y[te_idx],tx[te_idx],w) #TODO enter tester
#     if min_tuple[0]>te_loss:
#         min_tuple=(te_loss,param)
#     te_loss_log.append(te_loss)
#     tr_loss_log.append(tr_losses)
#     print(tr_losses,param)

# #train the regressor
# tr_idx,te_idx=helper.split_data(y,0.8)
# w,tr_loss=reg.logistic_regression(y[tr_idx],tx[tr_idx],50,50)
# te_loss=sp.logistic_log_likelihood(y[te_idx],tx[te_idx],w)
# #evaluate the result
# print(tr_loss,te_loss)

#saving the test
#helper.submit("logistic60","../test.csv",w,standard=True,islogistic=True)

# golden_list=[41, 71, 47, 37, 1, 13, 48, 45, 55, 50, 5, 52, 82, 7, 43, 19, 30, 60, 12, 44, 58, 31, 51, 42, 72, 54, 16, 81, 67, 22, 26, 56, 86, 57, 21, 0, 2, 62,
# 10, 64, 61, 33]
# def golden_input(tx,golden_list):
#     N=tx.shape[0]
#     F=tx.shape[1]
#     d=np.zeros((N,len(golden_list)))
#     for i in range(0,len(golden_list)):
#         num=golden_list[i]
#         d[:,i]=tx[:,num%F]**(int(num/F)+1)
#     return d
# gtx=golden_input(tx,golden_list)
gtx=degree(tx,2)
w,loss=reg.logistic_regression_gradient_descent_demo(y,gtx)
y,tx,ids=helper.load_csv_data("../test.csv",standard=True)
def degree_calc(tx,weight,degree):
    step=tx.shape[1]
    dotsum=0
    for i in range(0,degree):
        dotsum+=np.dot(tx**(i-1),weights[range(step*i,step*(i+1))])
    y_pred=1/(1+np.exp(-x))
    y_pred=1-y_pred
    return y_pred
pred=degree_calc(w,gtx,2)
helper.create_csv_submission(ids,pred,'JustTesting')

#Cascader
def pred_cascader(tx,ws,num=10):
    w_tx=np.ones([tx.shape[0],num+1])
    for n in range(0,num):
        x=np.dot(tx,ws[n])
        exp_x=np.exp(x)
        w_tx[:,n]=gen_pdf(1,x,exp_x)
    x=np.dot(w_tx,ws[num])
    exp_x=np.exp(x)
    return np.around(gen_pdf(1,x,exp_x))
def cascader(y,tx,num=10):
    w_tx=np.ones([tx.shape[0],num+1])
    ws=[]
    for n in range(0,num):
        w=gen_grad_regression(y,tx)
        ws.append(w)
        x=np.dot(tx,w)
        exp_x=np.exp(x)
        w_tx[:,n]=gen_pdf(y,x,exp_x)
        print(n)
    w=gen_grad_regression(y,w_tx)
    ws.append(w)
    return ws

#Feature selection
def forward_selection(y,tx,threshold=1e-4):
    used=[]
    losses=[]
    left=set(range(0,tx.shape[1]))
    for i in range(len(used),tx.shape[1]):
        best=(999,0)
        left=left-set(used)
        for j in left:
            tmp_use=used[:]
            tmp_use.append(j)
            w,loss=reg.least_squares(y,tx[:,tmp_use])
            if(best[0]>loss):
                best=(loss,tmp_use)
            #print("j={j}, loss={loss}, tmp_use={tmp_use}".format(j=j,loss=loss,tmp_use=tmp_use))
        used=best[1]
        losses.append(best[0])
        print("Selected number of F={f}, loss={loss}, err={err}".format(f=i,loss=best,err=helper.log_line(y,np.dot(tx[:,used],w),[1])[0]))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return used,losses
tx90=np.zeros((tx.shape[0],tx.shape[1]*3))
tx90[:,:tx.shape[1]]=tx
tx90[:,tx.shape[1]:(tx.shape[1]*2)]=tx**2
tx90[:,(tx.shape[1]*2):]=np.sin(tx)
#print(forward_selection(y,tx90))
used=[41, 61, 47, 60, 37, 48, 45, 55, 11, 73, 50, 4, 89, 52, 44, 58, 34, 22, 83, 63, 12, 0, 7, 3, 2, 32, 13, 67, 76, 43, 71, 54, 16, 21, 64, 82, 62, 57]
t=1
tx=tx90[:,used]
w,loss=reg.logistic_regression_bilguun(y,tx)
while t!=0:
    t=input("Enter:")
    line=helper.log_line(np.matrix(y).transpose(),np.dot(tx,w),[1],float(t))
    print(line)