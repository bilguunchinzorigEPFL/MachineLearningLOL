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

def load_csv_data(data_path, sub_sample=False, null_replace=0, standard=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (1,0).We assign 1 for label "b" and 0 otherwise
    yb = np.zeros(len(y))
    yb[np.where(y=='b')] = 1

    
    #converting NA's
    input_data[input_data == -999] = null_replace

    if standard:
        input_data=standardize(input_data)

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, threshold,islogistic=False):
    """Generates class predictions given weights, and a test data matrix"""
    if islogistic==True:
        x=np.dot(data,weights[:data.shape[1]])
        x=x+np.dot(data**2,weights[data.shape[1]:])
        y_pred=1/(1+np.exp(-x))
        y_pred=1-y_pred
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


def submit(name,test_path,weights,null_replace=0, standard=False,threshold=0.5,islogistic=False):

    #read data
    x = np.genfromtxt(test_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    data = x[:, 2:]
    #replacing null
    data[data == -999] = null_replace
    #standardize data
    if standard:
        data=standardize(data)
    predicted=predict_labels(weights,data,threshold,islogistic)
    create_csv_submission(ids,predicted,name)