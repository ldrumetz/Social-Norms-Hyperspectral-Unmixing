# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:42:35 2017

@author: administrateur
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def pca_viz(data,S):
        
    
   pca = PCA(n_components=3)
   scores = pca.fit_transform(np.transpose(data))
   
   U = pca.components_
   projected_first_argument = np.transpose(np.dot(U,data))   
   projected_second_argument = np.transpose(np.dot(U,S))   
   
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.plot(projected_first_argument[:,0], projected_first_argument[:,1], projected_first_argument[:,2], 'bo')
   ax.plot(projected_second_argument[:,0], projected_second_argument[:,1], projected_second_argument[:,2], 'ro')  
   # plt.gca().set_aspect('equal', adjustable='box')
   plt.draw()
   
   ax.set_xlabel('PC 1')
   ax.set_ylabel('PC 2')
   ax.set_zlabel('PC 3')