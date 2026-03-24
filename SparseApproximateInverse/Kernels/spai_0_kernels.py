from cupy import RawKernel
from pathlib import Path

spai_0_thread_smoother = RawKernel(
                         Path('SparseApproximateInverse/Kernels/SPAI_0_Threads_Kernel.cu').read_text(),
                      'spai_0_threads_kernel')
spai_0_thread_smoother.compile()

spai_0_warp_smoother = RawKernel(
                         Path('SparseApproximateInverse/Kernels/SPAI_0_Warp_Kernel.cu').read_text(),
                      'spai_0_warp_kernel')
spai_0_warp_smoother.compile()

spai_0_first_iteration_x0_0_smoother = RawKernel(
                         Path('SparseApproximateInverse/Kernels/SPAI_0_x0_0_Kernel.cu').read_text(),
                      'spai_0_first_iteration_x0_0')
spai_0_first_iteration_x0_0_smoother.compile()


def spai_0_thread_parallel_iteration(A,x,b,M,xnew):

    spai_0_thread_smoother(*spai_0_threads_config(A.shape[0]),
                      (A.shape[0], A.data, A.indptr, A.indices,
                            x, b, M.data,xnew))

    return xnew

def spai_0_warp_parallel_iteration(A,x,b,M,xnew):

    spai_0_warp_smoother(*spai_0_warp_config(A.shape[0]),
                    (A.shape[0], A.data, A.indptr, A.indices,
                          x, b, M.data,xnew))

    return xnew

def spai_0_first_iteration_x0_0(A,b,M,xnew):

    spai_0_first_iteration_x0_0_smoother(*spai_0_threads_config(A.shape[0]),
                                    (A.shape[0], b, M.data, xnew))

    return xnew


def spai_0_warp_config(SIZE):
    threads = 256
    warps_per_block = threads // 32
    blocks = (SIZE + warps_per_block - 1)//warps_per_block
    return (blocks,),(threads,)

def spai_0_threads_config(SIZE):
    threads = 256
    blocks = (SIZE + threads - 1)//threads
    return (blocks,),(threads,)

