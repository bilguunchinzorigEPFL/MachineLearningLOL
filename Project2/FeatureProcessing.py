import numpy as np
import re
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
        -1
def text2emb(text):
    words=text.replace("\n",'').split(" ")
    indices=[index(word) for word in words]
    indices=indices[indices>=0]
    words=embedding[indices]
    return np.sum(words,axis=1)
