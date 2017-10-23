import scripts.regressors as reg
import scripts.proj1_helpers as helper
import scripts.supportFunctions as sp
#load the data
y,tx,ids=helper.load_csv_data("data/train.csv",standard=True)

#Parameter testing
#params=[]
min_tuple=(999,0)
te_loss_log=[]
tr_loss_log=[]
param=10
while param!=0:
    param=float(input("Please enter parameter value: "))
    tr_idx,te_idx=helper.split_data(y,0.8)
    w,tr_losses=reg.logistic_regression(y[tr_idx],tx[tr_idx],50,param) #TODO enter regressor
    te_loss=sp.logistic_log_likelihood(y[te_idx],tx[te_idx],w) #TODO enter tester
    if min_tuple[0]>te_loss:
        min_tuple=(te_loss,param)
    te_loss_log.append(te_loss)
    tr_loss_log.append(tr_losses)
    print(tr_losses,param)

# #train the regressor
# tr_idx,te_idx=helper.split_data(y,0.8)
# w,tr_loss=reg.least_squares(y[tr_idx],tx[tr_idx])
# te_loss=sp.loss_mse(y[te_idx],tx[te_idx],w)
# #evaluate the result
# print(tr_loss,te_loss)

#saving the test
helper.submit("least square","../test.csv",w,standard=True)