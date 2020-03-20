This toolbox contains several scripts and functions in Python, to unmix hyperspectral data using the social norms, used to incorporate various flavors of sparsity accounting for the group structure of the endmember dictionary. This is useful to handle endmember variability in an easy and more robust way than classical approaches.

This code is associated to the article:

Drumetz, L., Meyer, T. R., Chanussot, J., Bertozzi, A. L., & Jutten, C.
(2019). Hyperspectral image unmixing with endmember bundles and group
sparsity inducing mixed norms. IEEE Transactions on Image Processing,
28(7), 3435-3450.

The contents include:

- demo_houston.py: example of use of the algorithms on a real hyperspectral dataset
- social_norms.py: the code of the unmixing algorithms, with three different possible social norms: 
	* group: inter group sparsity (at the material level)
	* elitist: intra group sparsity (at the atom level for each material)
	* fractional: both inter and intra group sparsity
- real_data_1.mat: real dataset used (crop of the DFC 2013 data)
- bundles.mat: endmember bundle used in the demo, and the associated group structure
- FCLSU.py : function performing the standard fully constrained least squared unmixing.
- pca_viz.py: function projecting data and endmembers on the space spanned by the first three principal components of the data and displaying a scatterplot
- rescale.py: function rescaling hyperspectral data between 0 and 1
- proj_simplex.py: function projecting one or several vectors onto the unit simplex

