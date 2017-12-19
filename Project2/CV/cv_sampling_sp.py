from sklearn.svm import SVC 
import numpy as np
import random
random.seed(9001)

def cv_data_loader(path,label):
    with open(path,'r') as target:
        rows=target.readlines()
        indices=np.asarray([np.r_[0.:len(rows)],np.floor(np.random.uniform(0,1,len(rows))*len(rows))])
        indices=np.transpose(indices)
        indices=np.argsort(indices[:,1]) # Estos son los indices de las filas que se tienen que quedar en la lista de tweets. 
        training1=np.int_(indices[0:80000])
        test1=np.int_(indices[80000:len(indices)])
        training_data1=[rows[training1]for training1 in training1 ]
        test_data1=[rows[test1]for test1 in test1 ]
        training2=np.transpose(np.int_(np.concatenate((np.transpose(indices[0:20000]),np.transpose(indices[40000:100000])))))
        test2=np.int_(indices[20000:40000])
        training_data2=[rows[training2]for training2 in training2 ]
        test_data2=[rows[test2]for test2 in test2 ]
        training3=np.transpose(np.int_(np.concatenate((np.transpose(indices[0:40000]),np.transpose(indices[60000:100000])))))
        test3=np.int_(indices[40000:60000])
        training_data3=[rows[training3]for training3 in training3 ]
        test_data3=[rows[test3]for test3 in test3 ]
        training4=np.transpose(np.int_(np.concatenate((np.transpose(indices[0:60000]),np.transpose(indices[80000:100000])))))
        test4=np.int_(indices[60000:80000])
        training_data4=[rows[training4]for training4 in training4 ]
        test_data4=[rows[test4]for test4 in test4 ]
        training5=np.int_(indices[20000:100000])
        test5=np.int_(indices[00000:20000])
        training_data5=[rows[training5]for training5 in training5 ]
        test_data5=[rows[test5]for test5 in test5 ]
        return [label]*len(rows),training_data1, test_data1, training_data2, test_data2,training_data3, test_data3,training_data4, test_data4, training_data5, test_data5   