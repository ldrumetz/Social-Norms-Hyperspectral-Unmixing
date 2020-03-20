"""
Created on Wed Nov 22 00:36:43 2017

@author: Lucas

Fully Constrained Least Squares Unmixing (FCLSU): unmix a hyperspectral image
using the positivity and sum to one constraints on the abundances

inputs: data: LxN data matrix (L: number of spectral bands, N: number of pixels)
        S0 : LxP reference endmember matrix (P: number of endmembers)
        
outputs: A: PxN abundance matrix
         
"""

import numpy as np
from proj_simplex import proj_simplex

def FCLSU(data,S0):
    
    [L,N] = data.shape   
    
    [L,P] = S0.shape 
    
    phi = np.ones([P,N])
    
    U = phi # split variable
    D = np.zeros(phi.shape) # Lagrange mutlipliers

    rho = 1

    maxiter_ADMM = 1000
    tol_phi = 10**-5

    S0tX = np.dot(np.transpose(S0),data)
    S0tS0 = np.dot(np.transpose(S0),S0)
    I = np.identity(P)

    for i in np.arange(maxiter_ADMM):
    
        phi_old = phi
    
        phi = np.dot(np.linalg.inv(S0tS0 + rho*I),S0tX + rho*(U-D))
    
#        U = np.maximum(U+D,0)
        U = proj_simplex(U+D)
    
        D = D + phi - U
    
        rel_phi = np.abs((np.linalg.norm(phi,'fro')-np.linalg.norm(phi_old,'fro'))/np.linalg.norm(phi_old,'fro'))
  
        print("iteration ",i," of ",maxiter_ADMM,", rel_phi =",rel_phi)

        if rel_phi < tol_phi :
            break



    A = phi
    
    return A