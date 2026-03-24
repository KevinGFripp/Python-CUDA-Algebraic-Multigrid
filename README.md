A Python + CUDA (cuPy) implementation of Algebraic Multigrid (AMG) for sparse linear systems of equations (Ax = b) discretised onto a 2D Cartesian grid.

A is expected to be diagonally dominant and of CSR format.

All data formats are expected to be single precision, in row-major order (ndarrays or csr_matrix).

AMG supports 'V' or 'F' cycles.

AMG can be used as a standalone iterative solver or as a preconditioner, implemented for bicgstab(1).


Required packages:

  cuPy,
  
  NumPy,
  
  Numba,
  
  SciPy,
  
  Matplotlib
