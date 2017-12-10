from sklearn.svm import SVC 
import Supports as sp
import FeatureProcessing as fp
import numpy as np

#Preparing the data
data=sp.data_loader("twitter-datasets/train_pos.txt",1)+sp.data_loader("twitter-datasets/train_neg.txt",-1)
train_i,valid_i,test_i=sp.split_data(data)
(p_data,labels)=[(fp.text2emb(text[0]),text[1]) for text in data]

#Training
model=SVC()
model.fit(p_data[train_i],labels[train_i])
print(np.mean(np.abs(labels[test_i]-model.predict(p_data[test_i]))))