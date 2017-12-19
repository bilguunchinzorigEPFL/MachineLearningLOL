from sklearn.svm import SVC 
import numpy as np
import random
import cv_sampling_sp as cv_sp
import pandas as pd
import re
from scipy.sparse import *
import pickle

# Upload the data 


label_p, data_p_cv1,test_p_cv1,data_p_cv2,test_p_cv2,data_p_cv3,test_p_cv3,data_p_cv4,test_p_cv4, data_p_cv5,test_p_cv5 =cv_sp.cv_data_loader("train_pos.txt",1)
label_n, data_n_cv1,test_n_cv1,data_n_cv2,test_n_cv2,data_n_cv3,test_n_cv3,data_n_cv4,test_n_cv4, data_n_cv5,test_n_cv5 =cv_sp.cv_data_loader("train_neg.txt",-1)

#training data 
f = open('data_p_cv5.txt', 'w')
f.writelines(["%s\n" % i  for i in data_p_cv5])

f2 = open('data_n_cv5.txt', 'w')
f2.writelines(["%s\n" % i  for i in data_n_cv5])   

data_cv5=data_p_cv5+data_n_cv5


# Testing data 
f = open('test_p_cv5.txt', 'w')
f.writelines(["%s\n" % i  for i in test_p_cv5])

f2 = open('test_n_cv5.txt', 'w')
f2.writelines(["%s\n" % i  for i in test_n_cv5])   

test_cv5=test_p_cv5+test_n_cv5

