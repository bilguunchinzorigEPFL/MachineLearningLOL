from sklearn.svm import SVC 
import Supports as sp
import FeatureProcessing as fp
import numpy as np

#If data is calculated
fp.proccess()
data,labels=fp.load_processed_data()
train_i,test_i=sp.split_data(data)
print("Splitted dataset:{tr},{te}".format(tr=len(train_i),te=len(test_i)))
print(data.shape)
print(type(labels))
#Training
model=SVC()
model.fit(data[train_i,:],labels[train_i])
print('Trained')
print(np.mean(np.abs(labels[test_i]-model.predict(data[test_i]))))

