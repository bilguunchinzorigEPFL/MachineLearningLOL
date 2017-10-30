import scripts.regressors as reg
import scripts.proj1_helpers as helper
import scripts.supportFunctions as sp
import numpy as np
from Probit import *
#load the data
y,tx,ids=helper.load_csv_data("data/train.csv",standard=True,normal=True)
print("Started")
degree=10000
tx=sp.build_poly(tx,degree)
lambda_=0.7
w_initial=np.zeros((tx.shape[1]))
weights, loss=reg.logistic_regression2(y,tx,max_iters,gamma,w_initial)
#w=reg.logistic_regression_bilguun(y, tx)
y,tx,ids=helper.load_csv_data("../test.csv",standard=True,normal=True)
tx=sp.build_poly(tx,degree)
y_pred=helper.predict_labels(weights, tx, threshold=0.5,islogistic=True)
name="logistic_monika"
helper.create_csv_submission(ids, y_pred, name)
print("end")

weights, loss=reg.logistic_regression_bilguun(y,tx)
#w=reg.logistic_regression_bilguun(y, tx)
y,tx,ids=helper.load_csv_data("../test.csv",standard=True,normal=True)
tx=sp.build_poly(tx,degree)
y_pred=helper.predict_labels(weights, tx, threshold=0.5,islogistic=True)
name="logistic_Beku"
helper.create_csv_submission(ids, y_pred, name)
print("end")