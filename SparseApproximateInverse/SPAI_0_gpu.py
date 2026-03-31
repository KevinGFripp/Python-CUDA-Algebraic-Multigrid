from cupyx.scipy.sparse import csr_matrix
from cupy import ndarray,empty_like
from SparseApproximateInverse.Kernels.spai_0_kernels import (spai_0_thread_parallel_iteration,
                                                             spai_0_warp_parallel_iteration)
from SparseApproximateInverse.Kernels.spai_0_kernels import (spai_0_first_iteration_x0_0,
                                                             spai_0_thread_parallel_colouring_iteration)
from SparseApproximateInverse.Kernels.spai_0_kernels import spai_0_first_iteration_x0_0_red_black

def spai_0_gpu_x0_0_smoother(A: csr_matrix,M: csr_matrix,b: ndarray,x0: ndarray,iterations: int):
    '''
    Computes the spai iterations x = x + M(b-Ax), with the optimisation for x0 =0 at the first iteration. Switch
    between thread based and warp-parallel kernels based upon the average number of non-zeros per row of A.
    :param A: csr_matrix
    :param M: csr_matrix, diagonal spai_0
    :param b: ndarray
    :param x0: ndarray
    :param iterations: int
    :return: solution: ndarray
    '''
    average_sparsity = A.nnz//A.shape[0]

    x = x0.copy()
    x_new = empty_like(x)

    #first iteration, assuming x0=0 requires no spMV operation
    x_new = spai_0_first_iteration_x0_0(A,b,M,x_new)

    # swap arrays
    x, x_new = x_new, x

    if average_sparsity < 15:

        for _ in range(1,iterations):
            x_new = spai_0_thread_parallel_iteration(A, x, b, M, x_new)
            #swap arrays
            x,x_new = x_new,x

    else:
        for _ in range(1,iterations):
            x_new = spai_0_warp_parallel_iteration(A, x, b, M, x_new)
            # swap arrays
            x, x_new = x_new, x


    if iterations % 2 == 0:
        return x
    else:
        return x_new


def spai_0_gpu_smoother(A: csr_matrix,M: csr_matrix,b: ndarray,x0: ndarray,iterations: int):
    '''
    Computes the spai iterations x = x + M(b-Ax).
        :param A: csr_matrix
        :param M: csr_matrix, diagonal spai_0
        :param b: ndarray
        :param x0: ndarray
        :param iterations: int
        :return: solution: ndarray
    '''

    average_sparsity = A.nnz//A.shape[0]

    x = x0.copy()
    x_new = empty_like(x)


    if average_sparsity < 15:

        for _ in range(iterations):
            x_new = spai_0_thread_parallel_iteration(A, x, b, M, x_new)
            #swap arrays
            x,x_new = x_new,x

    else:
        for _ in range(iterations):
            x_new = spai_0_warp_parallel_iteration(A, x, b, M, x_new)
            # swap arrays
            x, x_new = x_new, x


    if iterations % 2 == 0:
        return x
    else:
        return x_new


def spai_0_gpu_red_black_smoother(A: csr_matrix,M: csr_matrix,b: ndarray,x: ndarray,iterations: int, Nx: int, Ny: int):
    '''
    Computes the spai iterations x = x + M(b-Ax), using red-black smoothing. Perform one sweep over even
    (i+j % 2 == 0) (red) cells in the grid, followed by a second sweep (black), of odd cells (i+j % 2 == 1) to complete.
    This offers better smoothing than a pure Jacobi update due to partial incorporation of the current solution in the update.
    :param A:
    :param M:
    :param b:
    :param x:
    :param iterations:
    :param Nx:
    :param Ny:
    :return:
    '''
    for _ in range(iterations):
        spai_0_thread_parallel_colouring_iteration(A,Nx,Ny, x, b, M)

    return x



def spai_0_gpu_x0_0_red_black_smoother(A: csr_matrix,M: csr_matrix,b: ndarray,x0: ndarray,iterations: int, Nx: int, Ny: int):
    '''
    Computes the spai iterations x = x + M(b-Ax), using red-black smoothing. Perform one sweep over even
    (i+j % 2 == 0) (red) cells in the grid, followed by a second sweep (black), of odd cells (i+j % 2 == 1) to complete.
    This exploits the zero initial guess by saving a spMV operation, i.e. b-Ax0 = b.
    :param A:
    :param M:
    :param b:
    :param x0:
    :param iterations:
    :param Nx:
    :param Ny:
    :return:
    '''
    x = x0.copy()

    #first iteration, assuming x0=0 requires no spMV operation
    spai_0_first_iteration_x0_0_red_black(A,Nx,Ny, x, b, M)

    for _ in range(1,iterations):
        spai_0_thread_parallel_colouring_iteration(A,Nx,Ny, x, b, M)


    return x





