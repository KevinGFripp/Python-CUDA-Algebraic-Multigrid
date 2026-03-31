# Python + CUDA Algebraic Multigrid
High-performance GPU-accelerated multigrid / multigrid + bicgstab solver for structured Cartesian grids.

## Overview
Implementation of algebraic multigrid (AMG) as a standalone iterative solver, or preconditioner for bicgstab(1), aimed at solving 2D Poisson equations:

∇²u(x,y) = f(x,y) , 

or other diagonally dominant and structured sparse linear systems of equations, Ax=b.


## Multigrid Construction
- Bi-linear or cubic restriction/prolongation operators
- Row-weighted "Sparse Approximate Inverse" Jacobi smoother
- Red-Black colouring fine grid post-smoothing
- 'V' and 'F' cycles


## Features
- Flexibility of cuPy
- CUDA-accelerated fused stencil operations
- Warp-level and thread-level spMV CUDA kernel optimisations
- Supports arbitrary (even/odd) grid sizes
- Single precision
- CPU and GPU versions

## Structure

├── BICGSTAB_L/            # Pre-conditioned bicgstab(1) solver and kernels  

├── Multigrid/         # Base algbebraic multigrid implementation, cycles and kernels 

├── Laplacians/          # Example 2nd order Laplacian matrix 

├── SparseApproximateInverse/     # Multigrid smoother and kernels

Example_AMG_PoissonProblem.py # Solve the poisson problem on the CPU and GPU, and compare performance.

# Example : Poisson Problem

<img width="1200" height="500" alt="Solutions" src="https://github.com/user-attachments/assets/fcec1d2e-c467-49db-835a-2797e5bebd09" />


## 📊 Performance (AMD Ryzen 9 9950x3D vs Nvidia RTX 4090)


<img width="567" height="580" alt="Performance" src="https://github.com/user-attachments/assets/8129cd25-7797-4812-950f-0fef3e047829" />

### Takeaways
- The CPU is expected to be competitive at very small problem sizes.
- Below a grid size of $512^2$, kernel execution overhead is a dominant contribution to the solve time.
- At larger problem sizes, the GPU becomes saturated and kernel overhead is suppressed.
- At $4096^2$ cells, the GPU is ~45x faster than the CPU (80 ms per solve).

## Considerations

Strictly for a Laplacian matrix on a regular grid, the stencil may be known ahead of time. This would replace spMV operations by e.g. the smoother with much faster stencil CUDA kernels.

This whilst improving performance, however, would hard-code the problem. Keeping 'A' as a matrix makes it more flexible, for example, applied to non-uniform Cartesian grids, modified stencils and boundary conditions.

## Installation

### Requirements
- CUDA >= 11.0
- Python >= 3.10
- NumPy,
- Numba,
- CuPy

