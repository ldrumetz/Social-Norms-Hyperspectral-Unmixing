"""
Created on Wed Nov 22 00:36:43 2017

@author: Lucas

    We try to minimize the following cost function:
        J(A) = 1/2 * ||X - BA||_{F}^{2} + \lambda ||A||_{G,p,q}

    with B a collection of endmember candidates, A the
    abundances in each pixel and for each candidate.
    G is the group structure used, and p and q can be:
        (p,q) = (2,1) for the group penalty
        (p,q) = (1,2) for the elitist penalty
        (p,q) = (1,fraction) for the fractional penalty (0 < fraction <= 1)

    The abundances are subject to the usual nonnegativity and sum to one
    constraints.

Inputs:

    -data = LxN data matrix with L the number of spectral bands, and N the
    number of pixels. 
    -sources = LxQ endmember matrix, with Q the total number of endmember 
    candidates.
    -groups = Qx1 vector indicating the group structure of the abundance
    matrix. Values have to range between 1 and P, the number of groups 
    (endmembers).
    -A_init: initial QxN abundance matrix: especially useful for the 
    fractional case, where the optimization problem is not convex.
    -lambda: regularization parameter for the sparsity inducing terms
    -type: string indicating which penalty to use: 'group' for group lasso,
    'elitist' for elitist lasso, and 'fractional' for fractional lasso.
    -fraction: fraction to use for the fractional case (not used otherwise).
    0 < fraction <= 1 (if fraction = 1, this is a regular lasso, otherwise
    the problem becomes nonconvex but allows to use the sum to one constraint
    while promoting both intra and inter group sparsity)
    -maxiter_ADMM: maximum number of iterations before the algorithm stops.
    -tol_a : tolerance on the relative variation of the norm of the abundance
    matrix under which the algorithm is considered converged
    -verbose: flag for display in console. Display if true, no display
    otherwise 

Outputs: 
    -A: PxN abundance maps
    -optim_struct: structure containing the values of the cost function and
    its two terms (data fit and sparsity) for all the iterations.

reference: 

    Drumetz, L., Meyer, T. R., Chanussot, J., Bertozzi, A. L., & Jutten, C. 
    (2019). Hyperspectral image unmixing with endmember bundles and group 
    sparsity inducing mixed norms. IEEE Transactions on Image Processing,
    28(7), 3435-3450.

Author: Lucas Drumetz
Latest Revision: 16 March 2020
Revision: 1.4

         
"""

import numpy as np
from proj_simplex import proj_simplex

def social_unmixing(data,sources,groups,A_init,Lambda,rho,maxiter_ADMM,algo, \
                    fraction,tol_a, verbose):

    
    [L,N] = data.shape   
    Q = groups.shape[0]
    P = int(np.amax(groups))

    objective = np.zeros((maxiter_ADMM,1))
    norm_sparse = np.zeros((maxiter_ADMM,1))
    data_fit = np.zeros((maxiter_ADMM,1))
    
    A = np.ones((Q,N))
    
    U = np.copy(A) # split variable
    W = np.copy(A) # split variable
    D = np.zeros(A.shape) # Lagrange mutlipliers
    E = np.zeros(A.shape)

    StS = np.dot(sources.T,sources)
    StSp2rhoIp = StS +2*rho*np.identity(Q)


    if algo == 'group':
        
        V = np.copy(U);
        D = np.copy(E);
        
        for i in range(maxiter_ADMM):
    
            U_old = np.copy(U)
            U = np.dot(np.linalg.inv(StSp2rhoIp),np.dot(sources.T,data)+ rho*(V - D + W - E))
            V = prox_group_lasso(U+D,groups,Lambda/rho);

#           W = np.maximum(U+E,0)
            W = proj_simplex(U+E)

            D = D + U - V;
            E = E + U - W;

            group_norm = 0;
        
            for j in range(P):
                group_norm  = group_norm + np.sum((np.sqrt(np.sum(U[groups == j+1,:]**2))))
                
            objective[i] = 1/2 * np.linalg.norm(data-np.dot(sources,U),'fro')**2 + Lambda* group_norm
            norm_sparse[i] = Lambda * group_norm ;
            data_fit[i] = 1/2 * np.linalg.norm(data-np.dot(sources,U),'fro')**2
            
            rel_A = np.abs((np.linalg.norm(U,'fro')-np.linalg.norm(U_old,'fro'))/np.linalg.norm(U_old,'fro'))
  
            print("iteration ",i," of ",maxiter_ADMM,", rel_A =",rel_A)

            if i > 1 and rel_A < tol_a :
                break

    if algo == 'elitist':
          
        V = np.copy(U);
        D = np.copy(E);
        
        for i in range(maxiter_ADMM):
    
            U_old = np.copy(U)
            U = np.dot(np.linalg.inv(StSp2rhoIp),np.dot(sources.T,data)+ rho*(V + D + W + E))

            V = prox_elitist_group(U-D,groups,Lambda/rho);

#           W = np.maximum(U+E,0)
            W = proj_simplex(U-E)

            D = D - U + V;
            E = E - U + W;

            partial_elitist_norm = 0;
        
            for j in range(P):
                partial_elitist_norm = partial_elitist_norm + (np.sum(np.abs(U[groups == j+1,:])))**2;
                
            elitist_norm  = np.sum(np.sqrt(partial_elitist_norm))
                
            objective[i] = 1/2 * np.linalg.norm(data-np.dot(sources,U),'fro')**2 + Lambda* elitist_norm
            norm_sparse[i] = Lambda * elitist_norm ;
            data_fit[i] = 1/2 * np.linalg.norm(data-np.dot(sources,U),'fro')**2
            
            rel_A = np.abs((np.linalg.norm(U,'fro')-np.linalg.norm(U_old,'fro'))/np.linalg.norm(U_old,'fro'))
  
            print("iteration ",i," of ",maxiter_ADMM,", rel_A =",rel_A)

            if i > 1 and rel_A < tol_a :
                break

    if algo == 'fractional':
        M = np.zeros((P,Q))
        
        for i in range(P):
            M[i, groups == i] = 1
            
        U_old = U
        
        V = np.dot(M,U)
        D = np.zeros((P,N))
        MtM = np.dot(M.T,M)
        
        StspMtMp2rhoIp = (StS + rho* MtM + rho * np.identity(Q))
        
        for i in range(maxiter_ADMM):
            U = np.dot(np.linalg.inv(StspMtMp2rhoIp), np.dot(sources.T, data) + rho*np.dot(M.T,V+D) + rho *(W+E))
            V = approx_prox_fractional(np.dot(M,U)-D,Lambda/rho,fraction)
            W = proj_simplex(U-E)
            
            D = D - np.dot(M,U) + V
            E = E - U + W
#    
            fractional_norm = np.sum(np.sum(np.abs(U)**fraction))
            
            objective[i] = 1/2 * np.linalg.norm(data-np.dot(sources,U),'fro')**2 + Lambda* fractional_norm
            norm_sparse[i] = Lambda * fractional_norm ;
            data_fit[i] = 1/2 * np.linalg.norm(data-np.dot(sources,U),'fro')**2
            
            rel_A = np.abs((np.linalg.norm(U,'fro')-np.linalg.norm(U_old,'fro'))/np.linalg.norm(U_old,'fro'))
  
            print("iteration ",i," of ",maxiter_ADMM,", rel_A =",rel_A)

            if i > 1 and rel_A < tol_a :
                break

    A = np.copy(U)
    
    return A

def approx_prox_fractional(X,tau,p):
    Y = np.maximum(np.abs(X)-tau**(2-p)*np.abs(X)**(p-1),0)*np.sign(X);   
    return Y

def vector_soft_col(X,tau):
    NU = np.sqrt(np.sum(X**2))
    A = np.maximum(0, NU-tau)
    Y = np.tile(A/(A+tau),(np.shape(X)[0],1))* X
    return Y

def prox_elitist_group(X,groups,tau):
    nbg = int(np.amax(groups))
    [P,N] = np.shape(X)
    Y = np.zeros((P,N))
    for p in range(nbg):
        curr_X = X[groups == p+1,:]
        tau_p = tau/(1+tau) * np.sum(np.abs(curr_X), axis = 0)
        Y[groups == p+1,:] = np.maximum(np.abs(curr_X)-tau_p,0)*np.sign(curr_X)
    return Y
    
def prox_group_lasso(X,groups,tau):
    nbg = int(np.amax(groups))
    [P,N] = np.shape(X)
    Y = np.zeros((P,N))
    
    for p in range(nbg):
        Y[groups == p+1,:] = vector_soft_col(X[groups == p+1,:],tau)
    return Y