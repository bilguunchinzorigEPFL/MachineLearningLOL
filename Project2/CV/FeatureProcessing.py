import numpy as np
import re
import Supports as sp

def vocab_reader(path):
    voc=[]
    with open(path,'r') as file:
        return file.readlines()
vocabulary=vocab_reader("g_voc.txt")
embedding=np.load("g_emb.npy")

def index(word):
    try: 
        return vocabulary.index(word+"\n") 
    except: 
        return -1
def text2emb(text):
    words=text.replace("\n",'').split(" ")
    indices=[index(word) for word in words]
    indices=list(filter(lambda a: a >= 0, indices))
    words=embedding[indices]
    return np.sum(words,axis=0)/len(indices)

def load_processed_data():
    data=np.load('twitter-datasets\processedData.npy')
    label=[1]*100000+[-1]*100000
    return data,np.array(label)


def proccess():
    #Preparing the data
    #In this section we are loading the 
    data_p,label_p=sp.data_loader("twitter-datasets/train_pos.txt",1)
    data_n,label_n=sp.data_loader("twitter-datasets/train_neg.txt",-1)
    data=data_p+data_n
    p_data=np.array([text2emb(text) for text in data])
    print("Data loaded:{len},{shape}".format(len=len(p_data),shape=p_data.shape))
    p_data=np.nan_to_num(p_data)
    np.save('twitter-datasets\processedData',p_data)
    
