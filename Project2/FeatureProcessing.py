import numpy as np
import re
import scipy as sc

def vocab_reader(path):
    voc=[]
    with open(path,'r') as file:
        return file.readlines()
vocabulary=vocab_reader("g_voc.txt")
embedding=np.load("g_emb.npy")

def index(word):
    try: 
        vocabulary.index(word+"\n") 
    except: 
        return -1

def tweet_space(text):
    # Goal: For tweet t obtains features 
    words=text.replace("\n",'').split(" ")
    indices=[index(word) for word in words]
    indices=indices[indices>=0]
    words=embedding[indices]
    tweet_space=np.concatenate(np.sum(words,axis=0),np.max(words,axis=0), np.min(words,axis=0), np.std(words,axis=0), sc.skew(words,axis=0), axis=1)
    return tweet_space