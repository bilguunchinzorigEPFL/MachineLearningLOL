import csv
import numpy as np
import random
#create submission csv
def make_submission(name,y_pred):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(range(0,len(y_pred))+1, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
#load the tweets
def data_loader(path,label):
    data=[]
    with open(path,'r') as target:
        rows=target.readlines()
        return rows,[label]*len(rows)
#splits data into train,validation and test
def split_data(y, t_ratio=0.2, v_ratio=0.5, seed=1):
    train_i=random.sample(range(0,len(y)),int(len(y)*(1-t_ratio)))
    test_i=np.delete(range(0,len(y)),train_i,axis=0)
    return train_i,test_i