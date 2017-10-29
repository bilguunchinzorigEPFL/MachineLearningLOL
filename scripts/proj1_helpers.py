# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from scripts.supportFunctions import *

#splits given data into train and test data
def split_data(y, ratio=0.8, seed=1):
    np.random.seed(seed)
    train_i=np.random.choice(range(0,len(y)),size=int(len(y)*ratio))
    test_i=np.delete(range(0,len(y)),train_i,axis=0)
    return train_i,test_i
#standardize the data
def standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data
#normalize the data
def normalize(x,min_= -1,max_=1):
    return min_+((x-np.min(x))*(max_-min_))/(np.max(x)-np.min(x))
#null replacer
def null_replacer(x,null_val= -999):
    for c in range(0,x.shape[1]):
        total=0
        num=0
        nulls=[]
        for r in range(0,x.shape[0]):
            val=x[r,c]
            if val!=null_val:
                num+=1
                total+=val
            else: nulls.append(r)
        x[nulls,c]=total/num
    return x
def load_csv_data(data_path, sub_sample=False, null_replace=None, standard=False, normal=False, n_min= -1,n_max=1):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (1,0).We assign 1 for label "b" and 0 otherwise
    yb = np.zeros(len(y))
    yb[np.where(y=='b')] = 1
    #converting NA's
    if null_replace==None:
        input_data=null_replacer(input_data)
    else: input_data[input_data == -999] = null_replace
    if standard:
        input_data=standardize(input_data)
    if normal:
        input_data=normalize(input_data,n_min,n_max)
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, threshold=0.5,islogistic=False):
    """Generates class predictions given weights, and a test data matrix"""
    if islogistic==True:
        x=np.dot(data,weights)
        y_pred=1/(1+np.exp(-x))
    else:
	    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def log_saver(log,columns,name):
    with open(name, 'w') as file:
        fieldnames = columns
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r in log:
            writer.writerow(r)