A Python + CUDA (cuPy) implementation of Algebraic Multigrid (AMG) for sparse linear systems of equations (Ax = b) discretised onto a 2D Cartesian grid.

A is expected to be diagonally dominant and of CSR format.

All data formats are expected to be single precision, in row-major order (ndarrays or csr_matrix).

AMG supports 'V' or 'F' cycles.

AMG can be used as a standalone iterative solver or as a preconditioner, implemented for bicgstab(1).

Example Poisson solution:


<img width="1022" height="469" alt="Solutions" src="https://github.com/user-attachments/assets/a8266380-481f-46b7-a627-53d590339c19" />

<img width="487" height="462" alt="Performance" src="https://github.com/user-attachments/assets/533cebb8-b287-4d38-aa85-96b4a1a75bcd" />




Required packages: cuPy, NumPy, Numba, SciPy, Matplotlib
