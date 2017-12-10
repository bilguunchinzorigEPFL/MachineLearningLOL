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

# K-fold Cross-Validation: Parameter k in word embeddings 

from scipy.sparse import *
import numpy as np
import pickle
import random


def main():
    print("loading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save('embeddings', xs)


if __name__ == '__main__':
    main()


for i in range(3)
    embedding_dim=k[0,i]
    g_emb=
    model=SVC()
    model.fit(p_data[train_i],labels[train_i])
    print(np.mean(np.abs(labels[test_i]-model.predict(p_data[test_i]))))