from cupyx.scipy.sparse import csr_matrix
from cupy import ndarray,empty_like
from SparseApproximateInverse.Kernels.spai_0_kernels import spai_0_thread_parallel_iteration,spai_0_warp_parallel_iteration
from SparseApproximateInverse.Kernels.spai_0_kernels import spai_0_first_iteration_x0_0

def spai_0_gpu_x0_0_smoother(A: csr_matrix,M: csr_matrix,b: ndarray,x0: ndarray,iterations: int):
    '''
    Computes the spai iterations x = x + M(b-Ax), with the optimisation for x0 =0 at the first iteration.
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




