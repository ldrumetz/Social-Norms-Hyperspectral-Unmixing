# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 02:21:02 2017

@author: Lucas

rescales a nd array (and convert it to float) to use all the dynamic)

"""


import numpy as np

def rescale(data):
    
    maxi = np.amax(data)
    mini = np.amin(data)

    data_rescaled = (data.astype('float')-mini)/(maxi-mini)
 
    return data_rescaled