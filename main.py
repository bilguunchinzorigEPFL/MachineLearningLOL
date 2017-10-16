import scripts.regressors as reg
import scripts.proj1_helpers as helper
import scripts.supportFunctions as sp
#load the data
y,tx,ids=helper.load_csv_data("data/train.csv",standard=True)

#train the regressor
tr_idx,te_idx=helper.split_data(y,0.8)
w,tr_loss=reg.least_squares(y[tr_idx],tx[tr_idx])
te_loss=sp.loss_mse(y[te_idx],tx[te_idx],w)
#evaluate the result
print(tr_loss,te_loss)

#saving the test
#helper.submit("least square","../test.csv",w,standard=True)