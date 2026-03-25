# Python + CUDA Algebraic Multigrid
High-performance GPU-accelerated multigrid / multigrid + bicgstab solver for structured Cartesian grids.

## Overview
Implementation of algebraic multigrid (AMG) as a standalone iterative solver, or preconditioner for bicgstab(1), aimed at solving 2D Poisson equations:

∇²u(x,y) = f(x,y) , 

or other diagonally dominant and structured sparse linear systems of equations, Ax=b.


## Multigrid Construction
- Bi-linear or cubic restriction/prolongation operators
- Row-weighted "Sparse Approximate Inverse" Jacobi smoother
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

## 📊 Performance (AMD Ryzen 9 9950x3D vs Nvidia RTX 4090)
<img width="487" height="462" alt="Performance" src="https://github.com/user-attachments/assets/533cebb8-b287-4d38-aa85-96b4a1a75bcd" />

~ 37x versus the CPU at 4096^2 cells

## Installation

### Requirements
- CUDA >= 11.0
- Python >= 3.10
- NumPy,
- Numba,
- CuPy

