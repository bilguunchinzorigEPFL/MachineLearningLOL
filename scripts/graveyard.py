import scripts.regressors as reg
import scripts.proj1_helpers as helper
import scripts.supportFunctions as sp
import numpy as np
#load the data
y,tx,ids=helper.load_csv_data("data/train.csv",standard=True)

# tx60=np.zeros((tx.shape[0],tx.shape[1]*2))
# tx60[:,:tx.shape[1]]=tx
# tx60[:,tx.shape[1]:]=tx**2
# w=reg.logistic_regression_gradient_descent_demo(y, tx60)
# print("Started")
# def forward_selection(y,tx,threshold=1e-4):
#     used=[]
#     losses=[]
#     left=set(range(0,tx.shape[1]))
#     for i in range(0,tx.shape[1]):
#         best=(999,0)
#         left=left-set(used)
#         for j in left:
#             tmp_use=used[:]
#             tmp_use.append(j)
#             w,loss=reg.logistic_regression_gradient_descent_demo(y,tx[:,tmp_use])
#             if(best[0]>loss):
#                 best=(loss,tmp_use)
#             #print("j={j}, loss={loss}, tmp_use={tmp_use}".format(j=j,loss=loss,tmp_use=tmp_use))
#         used=best[1]
#         losses.append(best[0])
#         print("Selected number of F={f}, loss={loss}".format(f=i,loss=best))
#         if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
#             break
#     return used
def polynomial_maker(tx,degrees):
    step=tx.shape[1]
    result=np.zeros((tx.shape[0],step*len(degrees)))
    for i in range(0,len(degrees)):
        result[:,range(step*i,step*(i+1))]=tx**(degrees[i])
    return result
# print(forward_selection(y,degree(tx,3)))


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