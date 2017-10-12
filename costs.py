# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    e=y-np.dot(tx,w)
    lf=(1/len(y))*np.dot(np.transpose(e),e) 
    # raise NotImplementedError, how do I make this run 
    return lf
    
def compute_loss_mae(y, tx, w):
    e=y-np.dot(tx,w)
    iota=np.ones(len(y),)
    lf=(1/len(y))*np.dot(np.absolute(np.transpose(e)),iota)
    # raise NotImplementedError, how do I make this run 
    return lf