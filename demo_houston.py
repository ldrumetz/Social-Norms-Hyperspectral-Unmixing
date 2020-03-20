"""
This script reproduces the results of the algorithms presented in

Drumetz, L., Meyer, T. R., Chanussot, J., Bertozzi, A. L., & Jutten, C.
(2019). Hyperspectral image unmixing with endmember bundles and group
sparsity inducing mixed norms. IEEE Transactions on Image Processing,
28(7), 3435-3450.

to unmix a real dataset acquired above the University of Houston Campus,
in June 2012. We use a small part of the hyperspectral dataset provided
courtesy of the Hyperspectral Image Analysis group and the NSF Funded
Center for Airborne Laser Mapping at the University of Houston, and used
in the 2013 Data Fusion Contest (DFC):

  C. Debes et al., "Hyperspectral and LiDAR Data Fusion: Outcome of the
  2013 GRSS Data Fusion Contest," in IEEE Journal of Selected Topics in
  Applied Earth Observations and Remote Sensing, vol. 7, no. 6,
  pp. 2405-2418, June 2014.

Details on the dataset can also be found at:
http://hyperspectral.ee.uh.edu/?page_id=459

The blind unmixing is
performed using the group sparsity inducing norms tested in the paper
above, which incorporates spectral variability.
Several illustrative results are then displayed and quantitative metrics
are computed.

Author: Lucas Drumetz
Latest Revision: 17-March-2020
Revision: 1.4

DEMO group sparsity inducing norms
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from rescale import rescale
from social_unmixing import social_unmixing
from FCLSU import FCLSU
import h5py 
import time
from bundle2global import bundle2global
from pca_viz import pca_viz

plt.close("all")

data = scipy.io.loadmat("real_data_1.mat")

hyper = data['data']

rgb = hyper[:,:,[56,29,19]]

uint8_rgb = rescale(rgb)

plt.imshow(uint8_rgb, interpolation='nearest')


endmembers = {}
f = h5py.File('bundles.mat')
for k, v in f.items():
   endmembers[k] = np.array(v)
  
bundle = endmembers['bundle'].T

groups = endmembers['groups'][0].astype(int)

plt.figure()
plt.plot(bundle)

[m,n,L] = hyper.shape

P = int(np.amax(groups))
N = m*n;

imr = np.transpose(hyper.reshape((N,L),order='F'))

##

start = time.clock()
print('FLCSU bundle')
A_FCLSU_bundle = FCLSU(imr,bundle).T;
end = time.clock()
print(end - start)

#def social_unmixing(data,sources,groups,A_init,Lambda,rho,maxiter_ADMM,algo, \
#                    fraction,tol_a, verbose):

A_init = A_FCLSU_bundle
tol_a = 10**(-6)
rho = 10
fraction = 1/10
maxiter_ADMM = 1000

start = time.clock()
#algo = 'group'
#Lambda = 2

#algo = 'elitist'
#Lambda = 0.5

algo = 'fractional'
Lambda = 0.4


A_group = social_unmixing(imr,bundle, groups, A_init, Lambda,rho, maxiter_ADMM, algo, fraction, tol_a, True)
end = time.clock()
print(end - start)

A,sources_global = bundle2global(A_group,bundle,groups)

A_im =(np.transpose(A)).reshape((m,n,P),order = 'F')

plt.figure()

plt.subplot(151)
plt.imshow(A_im[:,:,0])
plt.xlabel('Concrete')
plt.subplot(152)
plt.imshow(A_im[:,:,1])
plt.xlabel('Red Roofs')
plt.subplot(153)
plt.imshow(A_im[:,:,2])
plt.xlabel('Asphalt')
plt.subplot(154)
plt.imshow(A_im[:,:,3])
plt.xlabel('Vegetation')
plt.subplot(155)
plt.imshow(A_im[:,:,4])
plt.xlabel('Structure')



pca_viz(imr,sources_global.reshape((L,P*N),order = 'F'))  