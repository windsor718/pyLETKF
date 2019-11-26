# pyletkf: LETKF python interface  
## Description  
pyletkf is a python interface for the data assimilation algorithm, Localized Transformed Ensemble Kalman Filter (Hunt et al., 2007).  
This module is especially desined for hydrological use, by natively supporting dynamic local patch generation along with a river network, whereas can be applied in any other fieled by defining local patches by yourself in a list format and saving as hdf5.  
LETKF part is implemented in optimized Cython language with static type declaration and memory views for numpy arrays to gain better performance. Multiprocessing is also supported for fast LETKF calculation.  
  
## Quick start  
See demos/demo_vector.ipynb for further description.  
  
## Notes  
2D grid based letkf is not supported in this version any more, due to the performance issue. Please vectorize your data into 1D and define nearest n local patch (traditional rectangular local patch) based on the vectorized data, and use mode="vector" when you call pyletkf. Although mode=grid still exists, calling mode="grid" may cause error. Those functions will be removed in future version.   
  
## Reference  
Hunt, B. R., E. J. Kostelich and I. Szunyogh, 2007: Efficient Data Assimilation for Spatiotemporal Chaos: A Local Ensemble Transform Kalman Filter. Physica D, 230, 112-126.
