import numpy as np
import re
<<<<<<< HEAD
import scipy as sc

=======
import Supports as sp
>>>>>>> a32f237980f20ef48f167a6efda28cd04da96ab9
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
<<<<<<< HEAD

def tweet_space(text):
    # Goal: For tweet t obtains features 
=======
def text2emb(text):
>>>>>>> a32f237980f20ef48f167a6efda28cd04da96ab9
    words=text.replace("\n",'').split(" ")
    indices=[index(word) for word in words]
    indices=list(filter(lambda a: a >= 0, indices))
    words=embedding[indices]
<<<<<<< HEAD
    tweet_space=np.concatenate(np.sum(words,axis=0),np.max(words,axis=0), np.min(words,axis=0), np.std(words,axis=0), sc.skew(words,axis=0), axis=1)
    return tweet_space
=======
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
    
>>>>>>> a32f237980f20ef48f167a6efda28cd04da96ab9
