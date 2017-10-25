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
helper.submit("logistic60","../test.csv",w,standard=True,islogistic=True)