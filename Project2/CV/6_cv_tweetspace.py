from sklearn.svm import SVC 
#import Supports as sp
#import FeatureProcessing as fp
import numpy as np
import cv_helpers as hp


# Training tweet space  

# Get the matrix of embeddings 

embeddings_cv=np.load("train_embeddings_cv2.npy")
vocabulary=hp.vocab_reader("vocab_cut_cv2.txt")
# Get the features or tweet space 
hp.proccess("data_p_cv2.txt","data_n_cv2.txt",embeddings_cv,'training_cv2',vocabulary)

# Testing tweet space  

# Get the matrix of embeddings 
test_embeddings_cv=np.load("test_embeddings_cv2.npy")
vocabulary_test=hp.vocab_reader("test_vocab_cut_cv2.txt")
# Get the features or tweet space 
hp.proccess("test_p_cv2.txt","test_n_cv2.txt",test_embeddings_cv,'testing_cv2',vocabulary_test)


