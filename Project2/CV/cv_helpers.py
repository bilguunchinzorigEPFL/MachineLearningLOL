from sklearn.svm import SVC 
import cv_supports as sp
import numpy as np

def vocab_reader(path):
    voc=[]
    with open(path,'r') as file:
        return file.readlines()

def index(word,vocabulary):
    try: 
        return vocabulary.index(word+"\n") 
    except: 
        return -1

def text2emb(text,embeddings,vocabulary):
    words=text.replace("\n",'').split(" ")
    indices=[index(word,vocabulary) for word in words]
    indices=list(filter(lambda a: a >= 0, indices))
    words=embeddings[indices]
    return np.sum(words,axis=0)/len(indices)

def load_processed_data(path):
    data=np.load(path)
    a=np.int(len(data)*0.5)
    label=[1]*a+[-1]*a
    return data,np.array(label)


def proccess(data_p,data_n,embeddings,path,vocabulary):
    #Preparing the data
    #In this section we are loading the 
    data_p,label_p=sp.data_loader(data_p,1)
    print(len(data_p))
    data_n,label_n=sp.data_loader(data_n,-1)
    print(len(data_n))
    data=data_p+data_n
    p_data=np.array([text2emb(text,embeddings,vocabulary) for text in data])
    print(p_data[0])
    print("Data loaded:{len},{shape}".format(len=len(p_data),shape=p_data.shape))
    p_data=np.nan_to_num(p_data)
    np.save(path,p_data)
    

