# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:07:40 2017

@author:  Lucas

This module performs the projection of the columns of a PxN matrix on the unit simplex (with P vertices).

Input: data matrix whose columns need to be projected on the simplex
Output: projected data matrix

"""

 # matlab code:: 
# proj_simplex_array = @(y) max(bsxfun(@minus,y,max(bsxfun(@rdivide,cumsum(sort(y,1,'descend'),1)-1,(1:size(y,1))'),[],1)),0); % projection on simplex

import numpy as np

def proj_simplex(data):

    data_sorted = np.sort(data,axis = 0)[::-1,:] # sort rows of data array in descending order (by going through each column backwards)
    cumulative_sum = np.cumsum(data_sorted,axis = 0)-1 # cumulative sum of each row
    vector = np.arange(np.shape(data_sorted)[0])+1 # define vector to be divided elementwise
    divided = cumulative_sum/vector[:,None] # perform the termwise division
    projected = np.maximum(data - np.amax(divided, axis = 0),np.zeros(divided.shape)) # projection step
    
    return projected
    
