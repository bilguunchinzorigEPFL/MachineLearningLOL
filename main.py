import scripts.regressors as reg
import scripts.proj1_helpers as helper
import scripts.supportFunctions as sp
import numpy as np
#load the data
y,tx,ids=helper.load_csv_data("data/train.csv",standard=True)
print("Started")
w=reg.grad_booster(y,tx)
tx=
y,tx,ids=helper.load_csv_data("../test.csv",standard=True)
pred=reg.pred_grad_booster(tx,w)
helper.create_csv_submission(ids,pred,'Somehow')