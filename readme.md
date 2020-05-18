# PyHMA: Python implementations of the Hydrograph Matching Algorithm
For introduction of the HMA see:
* Ewen, John. “Hydrograph Matching Method for Measuring Model Performance.” Journal of Hydrology 408, no. 1–2 (September 2011): 178–87. https://doi.org/10.1016/j.jhydrol.2011.07.038.

In summary, the article presents a method for identifying matching observed and simulated points in a way similar to how a hydrologist might compare to hydrographs. This is done by calculating (1) the 'work' for different (obs, sim) pairs, i.e. a distance measure based either on squared or absolute differences in magnitude and time and (2) identifying an optimal path to connect 0-2 simulation points to each observations that minimizes the cumulative work needed.

This package includes several implementations with different requirements on CPU time and RAM:
* calc_orig() follows exactly the pseudo-code from Ewen

The other implementations partially vectorize the computation, and:
* calc_sparse() uses sparse matrices for the cumulative work (CW) to reduce memory useage, at the cost of CPU time.
* calc_dense() uses dense matrices for CW to lower CPU time, but higher memory useage.
* calc_dense2() only maintains a single column of the CW matrix in memory, for best CPU time and RAM useage, but as a result the CW matrix is not available to check its accuracy and it is not possible to plot the connecting rays between sim and obs points.

These implementations also include support for non-constant time steps.

Note that all implementations except dense2 utilize a matrix of size (n, n, 2) for n observation time steps, so memory availability can become a limiting factor

See [readme.ipynb](https://github.com/icobro/PyHMA/blob/master/readme.ipynb) for demonstrations.
