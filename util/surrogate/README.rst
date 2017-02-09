Directory overview
------------------

- *SurrogateLogDet*: Class for approximating the logdet with
  a surrogate model. This class can be used with *estimate_params_gradient*.
- *SLHD*: Generates a symmetric Latin hypercube for a given
  number of design points in a specified number of dimensions.
- *best_slhd*: Generates multiple symmetric Latin hypercubes
  and returns the one that maximizes a convex combination of
  maximum minimum distance and the average correlation between
  variables.
- *CubicKernel*: A class for the cubic RBF kernel
- *TpsKernel*: A class for the Thin-plate spline RBF kernel
- *LinearTail*: A class for a linear polynomial tail
