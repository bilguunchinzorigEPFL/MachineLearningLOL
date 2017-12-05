import scripts.regressors as reg
import scripts.proj1_helpers as helper
import scripts.supportFunctions as sp
import numpy as np
#load the data
y,tx,ids=helper.load_csv_data("data/train.csv",standard=True)
print("Started")

# tx=sp.build_poly(tx,2)
# w=reg.grad_booster(y,tx)
# y,tx,ids=helper.load_csv_data("../test.csv",standard=True)
# pred=reg.pred_grad_booster(tx,w[0:30])
# pred+=reg.pred_grad_booster((tx**2)/1000,w[30:])
# helper.create_csv_submission(ids,pred,'Somehow')

# #1
# w,log=reg.least_squares_GD(y,tx,1000,0.1)
# helper.log_saver(log,['Err','Iteration','MSE'],'1-GD')

# #2
# w,log=reg.least_squares_SGD(y,tx,200,1000,0.1)
# helper.log_saver(log,['Err','Iteration','MSE'],'2-SGD')

# #3
# w,log=reg.least_squares(y,tx)
# helper.log_saver(log,['Err','MSE'],'3-LS',excep=True)

# #4
# log=[]
# lambdas=[x / 100 for x in range(0,1000)]
# for l in lambdas:
#     w,line=reg.ridge_regression(y,tx,l)
#     log.append(line+[l])
# helper.log_saver(log,['Err','MSE','lambda'],'4-LS reg')

# #5
# w,log=reg.logistic_regression_bilguun(y,tx)
# helper.log_saver(log,['Err','iter','Loss'],'5-Log')

# #6
# w,log=reg.logistic_regression_bilguun_reg(y,tx)
# helper.log_saver(log,['Err','iter','Loss'],'6-Log reg')

tx90=np.zeros((tx.shape[0],tx.shape[1]*3))
tx90[:,:tx.shape[1]]=tx
tx90[:,tx.shape[1]:(tx.shape[1]*2)]=tx**2
tx90[:,(tx.shape[1]*2):]=np.sin(tx)
#print(forward_selection(y,tx90))
used=[41, 61, 47, 60, 37, 48, 45, 55, 11, 73, 50, 4, 89, 52, 44, 58, 34, 22, 83, 63, 12, 0, 7, 3, 2, 32, 13, 67, 76, 43, 71, 54, 16, 21, 64, 82, 62, 57]
t=1
tx=tx90[:,used]
w,log=reg.grad_booster(y,tx)
print(w)
helper.log_saver(log,['iter','residual','err','loss'],'7-Boosting reg')