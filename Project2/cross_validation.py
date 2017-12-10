from sklearn.svm import SVC 
import Supports as sp
import FeatureProcessing as fp
import numpy as np
from scipy.sparse import *
import pickle
import random


# K-fold Cross-Validation: Parameter k in word embeddings 

embedding_dimension=[20, 50, 80, 100 ]

for i in range(3)

 # 1. Get the embedding with corresponding k 

    def main():
        print("loading cooccurrence matrix")
        with open('cooc.pkl', 'rb') as f:
            cooc = pickle.load(f)
        print("{} nonzero entries".format(cooc.nnz))

        nmax = 100
        print("using nmax =", nmax, ", cooc.max() =", cooc.max())

        print("initializing embeddings")
        embedding_dim = embedding_dimension[0,i]
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
        np.save('embeddings'+str(i), xs)

    if __name__ == '__main__':
        main()

    embedding=np.load("'embeddings'+str(i)")

    # 2. Create the tweet-space with embedings with k dimension 

    fn='twitter_dataset/neg_train.txt'
    with open(fn,'r') as ff:
        rows=ff.readlines()
        space_neg=[tweet_space(row) for row in rows]

    fn='twitter_dataset/pos_train.txt'
    with open(fn,'r') as ff:
        rows=ff.readlines()
        space_pos=[tweet_space(row) for row in rows]

    space=np.concatenate((space_neg,space_pos),axis=0)

    means=[]
    with open(fn,'r') as ff:
        rows=ff.readlines()
        space=[tweet_space(row) for row in rows]   
        model=SVC()
        model.fit(p_data[train_i],labels[train_i])
        means.append(np.mean(np.abs(labels[test_i]-model.predict(p_data[test_i]))))
        np.array(means)
        print(means)
        