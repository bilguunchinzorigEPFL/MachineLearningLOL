import csv
import numpy as np
import random
def make_submission(name,y_pred):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(range(0,len(y_pred))+1, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def data_loader(path,label):
    data=[]
    with open(path,'r') as target:
        rows=target.readlines()
        data=zip(rows,[label]*len(rows))
    return list(data)

def split_data(y, t_ratio=0.2, v_ratio=0.5, seed=1):
    np.random.seed(seed)
    train_i=np.random.choice(range(0,len(y)),size=int(len(y)*(1-t_ratio)))
    test_i=np.delete(range(0,len(y)),train_i,axis=0)
    valid_i=np.random.choice(test_i, size=int(len(test_i)*v_ratio))
    test_i=np.delete(range(0,len(y)),valid_i,axis=0)
    return train_i,valid_i,test_i