from cupy import RawKernel
from pathlib import Path

residual_threads_kernel = RawKernel(Path('BICGSTAB_L/Kernels/Residual_Threads_Kernel.cu').read_text(),
                        'residual_threads_kernel')
residual_threads_kernel.compile()

residual_warp_kernel = RawKernel(Path('BICGSTAB_L/Kernels/Residual_Warp_Kernel.cu').read_text(),
                        'residual_warp_kernel')
residual_warp_kernel.compile()

def residual_warp_config(SIZE):
    '''
    1D blocks of threads of size 256
    :param SIZE: int
    :return: (blocks,),(threads,)
    '''
    threads = 256
    warps_per_block = threads // 32
    blocks = (SIZE + warps_per_block - 1)//warps_per_block
    return (blocks,),(threads,)


def residual_threads_config(SIZE):
    '''
    1D blocks of threads of size 256
    :param SIZE: int
    :return: (blocks,),(threads,)
    '''
    threads = 256
    blocks = (SIZE + threads - 1)//threads
    return (blocks,),(threads,)


def compute_residual(A,x,b,r):
    '''
    Fused kernel computing r = b - Ax, switching between threads-based and warp-based kernels
    depending upon the average number of non-zero elements per row of A.
    :param A: csr_matrix
    :param x: ndarray
    :param b: ndarray
    :param r: ndarray
    :return: r: ndarray
    '''
    average_sparsity = A.nnz // A.shape[0]

    if average_sparsity < 15:
        residual_threads_kernel(*residual_threads_config(A.shape[0]),
                           (A.shape[0],
                                A.data,
                                A.indptr,
                                A.indices,
                                x, b, r))
    else:
        residual_warp_kernel(*residual_warp_config(A.shape[0]),
                       (A.shape[0],
                            A.data,
                            A.indptr,
                            A.indices,
                            x, b, r))

    return r