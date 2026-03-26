from selectors import SelectSelector

from Multigrid.Grid_gpu import Grid_GPU
from cupy import ndarray
from cupy.linalg import norm
from cupyx.scipy.sparse import csr_matrix
from SparseApproximateInverse.SPAI_0_gpu import spai_0_gpu_smoother,spai_0_gpu_x0_0_smoother
from Multigrid.Kernels.residual_restriction_kernels import restriction
from Multigrid.Kernels.prolongation_kernels import prolongation
from BICGSTAB_L.Kernels.residual_kernels import compute_residual

def amg_gpu(grids: list[Grid_GPU],b: ndarray,x0: ndarray,max_iterations: int,tol,cycle):
    '''
            GPU-based standalone AMG iterative solver.

            :param grids: list of Grid objects for AMG
            :param b: numpy ndarray
            :param x0: numpy ndarray
            :param max_iterations: int
            :param tol: float
            :param cycle: Multigrid cycle, 'V' or 'F'
            :return: x: solution ,r_norm: last iteration residual, iterations: total number of iterations
            '''
    ## setup
    numgrids = len(grids)

    if not tol:
        tol = 1e-6

    if not max_iterations:
        max_iterations = 20

    if cycle != 'V' or cycle != 'F':
        cycle = 'V'

    if cycle == 'V':
        f_cycles = 0
    else:
        f_cycles = numgrids -2

    # update RHS and initial solution
    grids[0].x = x0
    grids[0].b = b
    norm_b = norm(b)

    # iteration count
    iteration = 0
    ##

    ## Multigrid cycle
    for i in range(max_iterations):
        grids[0].x = amg_cycle(grids,cycle)

        iteration += 1

        # error
        grids[0].temp_array = residual(grids[0].Matrix,grids[0].x,grids[0].b)
        r_norm = (norm(grids[0].temp_array)/norm_b)

        if r_norm.get() <= tol:
            break


    return grids[0].x, r_norm, iteration

def amg_cycle(grids: list[Grid_GPU],cycle):
    '''
                CPU AMG cycle.

                :param grids: list of Grid objects for AMG
                :param cycle: Multigrid cycle, 'V' or 'F'
                :return: x: solution
                '''
    ## setup
    numgrids = len(grids)

    if cycle != 'V' or cycle != 'F':
        cycle = 'V'

    if cycle == 'V':
        f_cycles = 0
    else:
        f_cycles = numgrids - 2

    # fine grid
    # pre-smooth
    grids[0].x = spai_0_gpu_smoother(grids[0].Matrix,
                                     grids[0].M,
                                     grids[0].b,
                                     grids[0].x,
                                     iterations=2)

    # restrict to coarser mesh
    grids[1].b = restriction(
        compute_residual(grids[0].Matrix,
                         grids[0].x,
                         grids[0].b,
                         grids[0].temp_array),
        grids[1].b,
        grids[0].Nx, grids[0].Ny,
        grids[1].Nx, grids[1].Ny)

    # restriction
    for v in range(1, numgrids - 1, 1):
        # smooth error equation A *xn = r
        grids[v].x = spai_0_gpu_x0_0_smoother(grids[v].Matrix,
                                              grids[v].M,
                                              grids[v].b,
                                              grids[v].x,
                                              iterations=2)

        # restrict to coarser mesh
        grids[v + 1].b = restriction(
            compute_residual(grids[v].Matrix,
                             grids[v].x, grids[v].b,
                             grids[v].temp_array),
            grids[v + 1].b,
            grids[v].Nx, grids[v].Ny,
            grids[v + 1].Nx, grids[v + 1].Ny)

    # coarsest mesh
    grids[numgrids - 1].x = spai_0_gpu_x0_0_smoother(grids[numgrids - 1].Matrix,
                                                     grids[numgrids - 1].M,
                                                     grids[numgrids - 1].b,
                                                     grids[numgrids - 1].x,
                                                     iterations=2)

    # F cycle loop
    for F in range(0, f_cycles):
        # start on lowest grid, prolongate from coarsest grid to numgrids - F -2
        # prolongate
        for v in range(numgrids - 2, numgrids - F - 1, -1):
            # interpolate solution
            prolongation(grids[v].x, grids[v + 1].x,
                         grids[v].Nx, grids[v].Ny,
                         grids[v + 1].Nx, grids[v + 1].Ny)

            # post-smooth
            grids[v].x = spai_0_gpu_smoother(grids[v].Matrix,
                                             grids[v].M,
                                             grids[v].b,
                                             grids[v].x,
                                             iterations=1)

        # restrict back from numgrids - F to coarsest grid
        # restriction
        for v in range(numgrids - F - 1, numgrids - 1, 1):
            # smooth error equation A *xn = r
            grids[v].x = spai_0_gpu_x0_0_smoother(grids[v].Matrix,
                                                  grids[v].M,
                                                  grids[v].b,
                                                  grids[v].x,
                                                  iterations=2)

            # restrict to coarser mesh
            grids[v + 1].b = restriction(
                compute_residual(grids[v].Matrix,
                                 grids[v].x,
                                 grids[v].b,
                                 grids[v].temp_array),
                grids[v + 1].b,
                grids[v].Nx, grids[v].Ny,
                grids[v + 1].Nx, grids[v + 1].Ny)

        # solve on coarsest mesh
        grids[numgrids - 1].x = spai_0_gpu_x0_0_smoother(grids[numgrids - 1].Matrix,
                                                         grids[numgrids - 1].M,
                                                         grids[numgrids - 1].b,
                                                         grids[numgrids - 1].x,
                                                         iterations=2)
    # end of interior F cycle

    # prolongate
    for v in range(numgrids - 2, -1, -1):
        # interpolate solution
        prolongation(grids[v].x, grids[v + 1].x,
                     grids[v].Nx, grids[v].Ny,
                     grids[v + 1].Nx, grids[v + 1].Ny)

        # post-smooth
        grids[v].x = spai_0_gpu_smoother(grids[v].Matrix,
                                         grids[v].M,
                                         grids[v].b,
                                         grids[v].x,
                                         iterations=2)

    return grids[0].x




def amg_cycle_preconditioner(grids: list[Grid_GPU],b: ndarray,x0: ndarray,cycle):
    '''
                    CPU AMG cycle optimised as a preconditioner.

                    :param grids: list of Grid objects for AMG
                    :param x0: ndarray, initial solution vector, equal to 0
                    :param cycle: Multigrid cycle, 'V' or 'F'
                    :return: x: solution
    '''
    ## setup
    numgrids = len(grids)

    if cycle != 'V' or cycle != 'F':
        cycle = 'V'

    if cycle == 'V':
        f_cycles = 0
    else:
        f_cycles = numgrids - 2

    # update RHS and initial solution
    grids[0].x = x0
    grids[0].b = b

    ## Multigrid cycle
    grids[0].x = spai_0_gpu_x0_0_smoother(grids[0].Matrix,grids[0].M,grids[0].b,grids[0].x,iterations=2)

    # restrict to coarser mesh
    grids[1].b = restriction(
                             compute_residual(grids[0].Matrix,
                                              grids[0].x,
                                              grids[0].b,
                                              grids[0].temp_array),
                             grids[1].b,
                             grids[0].Nx, grids[0].Ny,
                             grids[1].Nx, grids[1].Ny)


    # restriction
    for v in range(1, numgrids - 1, 1):
        # smooth error equation A *xn = r
        grids[v].x = spai_0_gpu_x0_0_smoother(grids[v].Matrix,grids[v].M,grids[v].b,grids[v].x,iterations=2)

        # restrict to coarser mesh
        grids[v + 1].b = restriction(
                                     compute_residual(grids[v].Matrix,
                                                      grids[v].x, grids[v].b,
                                                      grids[v].temp_array),
                                     grids[v + 1].b,
                                     grids[v].Nx, grids[v].Ny,
                                     grids[v + 1].Nx, grids[v + 1].Ny)


    # coarsest mesh
    grids[numgrids - 1].x = spai_0_gpu_x0_0_smoother(grids[numgrids - 1].Matrix,
                                          grids[numgrids - 1].M,
                                          grids[numgrids - 1].b,
                                          grids[numgrids - 1].x,
                                          iterations=2)

    # F cycle loop
    for F in range(0, f_cycles):
        # start on lowest grid, prolongate from coarsest grid to numgrids - F -2
        # prolongate
        for v in range(numgrids - 2, numgrids - F - 1, -1):
            # interpolate solution
            prolongation(grids[v].x, grids[v + 1].x,
                         grids[v].Nx, grids[v].Ny,
                         grids[v + 1].Nx, grids[v + 1].Ny)

            # post-smooth
            grids[v].x = spai_0_gpu_smoother(grids[v].Matrix,grids[v].M,grids[v].b,grids[v].x,iterations=2)

        # restrict back from numgrids - F to coarsest grid
        # restriction
        for v in range(numgrids - F - 1, numgrids - 1, 1):
            # smooth error equation A *xn = r
            grids[v].x = spai_0_gpu_x0_0_smoother(grids[v].Matrix,grids[v].M,grids[v].b,grids[v].x,iterations=2)

            # restrict to coarser mesh
            grids[v + 1].b = restriction(
                                         compute_residual(grids[v].Matrix,
                                                          grids[v].x,
                                                          grids[v].b,
                                                          grids[v].temp_array),
                                         grids[v + 1].b,
                                         grids[v].Nx, grids[v].Ny,
                                         grids[v + 1].Nx, grids[v + 1].Ny)

        # solve on coarsest mesh
        grids[numgrids - 1].x = spai_0_gpu_x0_0_smoother(grids[numgrids - 1].Matrix,
                                                         grids[numgrids - 1].M,
                                                         grids[numgrids - 1].b,
                                                         grids[numgrids - 1].x,
                                                         iterations=2)
    # end of interior F cycle

    # prolongate
    for v in range(numgrids - 2, -1, -1):
        # interpolate solution
        prolongation(grids[v].x,grids[v + 1].x,
                     grids[v].Nx,grids[v].Ny,
                     grids[v + 1].Nx, grids[v + 1].Ny)

        # post-smooth
        grids[v].x = spai_0_gpu_smoother(grids[v].Matrix, grids[v].M, grids[v].b, grids[v].x,iterations=2)


    return grids[0].x


def restrict(R: csr_matrix,x: ndarray):
    return R @ x


def prolongate(I: csr_matrix, x: ndarray):
    return I @ x


def residual(A: csr_matrix,x: ndarray, b: ndarray):

    return b - A @ x