from Laplacians.Laplacian2D import laplacian_2d
import numpy as np
import matplotlib.pyplot as plt
from Multigrid.HierarchyOfGrids import hierarchy_of_grids
from BICGSTAB_L.AMG_BICGSTAB_L import amg_bicgstab_l as amg_bicgstab_cpu
from BICGSTAB_L.AMG_BICGSTAB_L_GPU import amg_bicgstab_l_gpu as amg_bicgstab
from Multigrid.GridsToGPU import grids_to_gpu
from cupy import asarray
import time as timer


if __name__ == '__main__':
    ## Solve del2 * x = b on a regular grid
    ## Comparison between CPU performance and GPU performance

    #config
    runs = 100
    tol = 1e-5

    # grid size
    Nx = 512
    Ny = 512

    ##setup
    # make 2D laplacian matrix
    del2 = laplacian_2d(Nx, Ny)

    x0 = np.zeros((Nx * Ny), dtype=np.float32)

    # rhs with 1s on top boundary
    b = np.zeros((Nx * Ny), dtype=np.float32)
    b[0:Ny] = -1.0

    tic = timer.time()

    multigrid, numgrids = hierarchy_of_grids(del2,b,x0,Nx,Ny)
    toc = timer.time()
    print('Setup time = ',toc-tic,' s')


    print('--Multigrid levels--')
    [print(multigrid[n].Nx,'x',multigrid[n].Ny) for n in range(numgrids)]
    print('--------------------')

    # move to GPU
    x0_gpu = asarray(x0, dtype=np.float32)
    b_gpu = asarray(b, dtype=np.float32)
    multigrid_gpu = grids_to_gpu(multigrid)

    #CPU testing
    tic = timer.time()

    for _ in range(runs):
        (solution_cpu,
         r_norm_cpu,
         iterations_cpu) = amg_bicgstab_cpu(multigrid,b,x0,max_iterations=10,
                                            tol=tol,cycle='V')

    toc = timer.time()
    print('CPU time = ', toc - tic, ' s')
    print('relative residual = ' + str(r_norm_cpu))
    print('iterations = ' + str(iterations_cpu))


    #GPU testing
    tic = timer.time()
    for _ in range(runs):
        (solution_gpu,
         r_norm_gpu,
         iterations_gpu) = amg_bicgstab(multigrid_gpu,b_gpu,x0_gpu,max_iterations=10,
                                        tol=tol,cycle='V')

    toc = timer.time()
    print('GPU time = ', toc-tic,' s')
    print('relative residual = ' + str(r_norm_gpu.get()))
    print('iterations = ' + str(iterations_gpu))


    plt.subplot(1,2,1)
    plt.imshow(solution_gpu.get().reshape((Nx,Ny)), cmap='turbo', interpolation='none',aspect="equal")
    plt.xlabel('x (cells)')
    plt.ylabel('y (cells)')
    plt.title('CUDA Solution')

    plt.subplot(1,2,2)
    plt.imshow(solution_cpu.reshape((Nx, Ny)), cmap='turbo', interpolation='none', aspect="equal")
    plt.xlabel('x (cells)')
    plt.ylabel('y (cells)')
    plt.title('CPU Solution')

    plt.show()





























