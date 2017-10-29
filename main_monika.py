import scripts.regressors as reg
import scripts.proj1_helpers as helper
import scripts.supportFunctions as sp
import numpy as np
#load the data
y,tx,ids=helper.load_csv_data("data/train.csv",standard=True,normal=True)
print("Started")
degree=2
tx=sp.build_poly(tx,degree)
max_iters=2
gamma=0.8
w_initial=np.zeros((tx.shape[1]))
weights=reg.logistic_regression2(y,tx,max_iters,gamma,w_initial)
print(weights.shape)
#w=reg.logistic_regression_bilguun(y, tx)
#y,tx,ids=helper.load_csv_data("../test.csv",standard=True,normal=True)
#y_pred=helper.predict_labels(weights, tx, threshold=0.5,islogistic=True)
#print(y_pred)
#name="logistic_monika"
#helper.create_csv_submission(ids, y_pred, name)