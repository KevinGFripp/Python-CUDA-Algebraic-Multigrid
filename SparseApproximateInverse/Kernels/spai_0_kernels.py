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



spai_0_thread_red_smoother = RawKernel(
                         Path('SparseApproximateInverse/Kernels/SPAI_0_Threads_Red_Kernel.cu').read_text(),
                      'spai_0_threads_red_kernel')
spai_0_thread_red_smoother.compile()

spai_0_thread_black_smoother = RawKernel(
                         Path('SparseApproximateInverse/Kernels/SPAI_0_Threads_Black_Kernel.cu').read_text(),
                      'spai_0_threads_black_kernel')
spai_0_thread_black_smoother.compile()


spai_0_x0_0_thread_red_smoother = RawKernel(
                         Path('SparseApproximateInverse/Kernels/SPAI_0_x0_0_Red_Kernel.cu').read_text(),
                      'spai_0_x0_0_threads_red_kernel')
spai_0_x0_0_thread_red_smoother.compile()

spai_0_x0_0_thread_black_smoother = RawKernel(
                         Path('SparseApproximateInverse/Kernels/SPAI_0_x0_0_Black_Kernel.cu').read_text(),
                      'spai_0_x0_0_threads_black_kernel')
spai_0_x0_0_thread_black_smoother.compile()


def spai_0_thread_parallel_colouring_iteration(A,Nx,Ny,x,b,M):

    spai_0_thread_red_smoother(*spai_0_threads_colouring_config(Nx,Ny),
                          (Nx,Ny, A.data, A.indptr, A.indices,
                                x, b, M.data))

    spai_0_thread_black_smoother(*spai_0_threads_colouring_config(Nx,Ny),
                            (Nx, Ny, A.data, A.indptr, A.indices,
                                  x, b, M.data))

    return


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

def spai_0_first_iteration_x0_0_red_black(A,Nx,Ny,xnew,b,M):

    spai_0_x0_0_thread_red_smoother(*spai_0_threads_colouring_config(Nx, Ny),
                               (Nx, Ny, xnew, b, M.data))

    spai_0_x0_0_thread_black_smoother(*spai_0_threads_colouring_config(Nx, Ny),
                               (Nx, Ny, A.data, A.indptr, A.indices,
                                xnew, b, M.data))

    return


def spai_0_warp_config(SIZE):
    threads = 256
    warps_per_block = threads // 32
    blocks = (SIZE + warps_per_block - 1)//warps_per_block
    return (blocks,),(threads,)

def spai_0_threads_config(SIZE):
    threads = 256
    blocks = (SIZE + threads - 1)//threads
    return (blocks,),(threads,)

def spai_0_threads_colouring_config(Nx: int, Ny: int):
    threads = 16
    blocks_x = (Ny + threads - 1)//threads
    blocks_y = (Nx + threads - 1)//threads

    return (blocks_x,blocks_y,),(threads,threads,)

