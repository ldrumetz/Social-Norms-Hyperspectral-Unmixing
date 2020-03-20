#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function sums abundance maps corresponding to the same group and 
computes pixelwise endmembers from the variant abundances

Inputs:
- A_bundle is the Q*N abundance matrix of all considered signatures
- bundle is the LxQ matrix containing these signatures
- groups is the Qx1 vector indicating the group structure of the abundance
matrix. Values have to range between 1 and P, the number of groups 
(endmembers).

Author: Lucas Drumetz
Latest Revision: 17-March-2019
Revision: 1.3
"""
import numpy as np


def bundle2global(A_bundle,bundle,groups):

    N = np.shape(A_bundle)[1]
    L = np.shape(bundle)[0]
    
    nbg = int(np.amax(groups))

    A_global = np.zeros((nbg,N))
    sources_global = np.zeros((L,nbg,N));


    threshold = 10**(-2);

# test

    A_bundle_new = np.copy(A_bundle)
    A_bundle_new[np.abs(A_bundle) < threshold] = 0;

    for p in range(nbg):
        A_global[p,:] = np.sum(A_bundle_new[groups == p+1,:],axis =0);

    
        for i in range(N):
            if A_global[p,i] !=0:
                sources_global[:,p,i] = np.sum(np.tile(np.expand_dims(A_bundle_new[groups == p+1,i],axis = 1),(1,L)).T \
                *bundle[:,groups ==p+1],axis = 1)/A_global[p,i]
            else:
                sources_global[:,p,i] = np.mean(bundle[:,groups == p+1],axis = 1)

    return A_global,sources_global
