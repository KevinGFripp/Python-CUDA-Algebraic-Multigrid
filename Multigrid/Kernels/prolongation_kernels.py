from cupy import RawKernel
from pathlib import Path
from cupy import ndarray

# prolongation_threads_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_Threads_Kernel.cu').read_text(),
#                         'prolongation_threads_kernel')
# prolongation_threads_kernel.compile()
#
# prolongation_warp_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_Warp_Kernel.cu').read_text(),
#                         'prolongation_warp_kernel')
# prolongation_warp_kernel.compile()


prolongation_2d_3pt_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_2D_3pt_Kernel.cu').read_text(),
                        'Prolongation_2D_NyNx_3pt')
prolongation_2d_3pt_kernel.compile()

prolongation_2d_4pt_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_2D_4pt_Kernel.cu').read_text(),
                        'Prolongation_2D_NyNx_4pt')
prolongation_2d_4pt_kernel.compile()

prolongation_2d_3pt_4pt_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_2D_3pt_4pt_Kernel.cu').read_text(),
                        'Prolongation_2D_NyNx_3pt_4pt')
prolongation_2d_3pt_4pt_kernel.compile()

prolongation_2d_4pt_3pt_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_2D_4pt_3pt_Kernel.cu').read_text(),
                        'Prolongation_2D_NyNx_4pt_3pt')
prolongation_2d_4pt_3pt_kernel.compile()





prolongation_2d_3pt_gather_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_2D_Gather_3pt_Kernel.cu').read_text(),
                        'Prolongation_2D_Gather_3pt_Kernel')
prolongation_2d_3pt_gather_kernel.compile()

prolongation_2d_3pt_4pt_gather_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_2D_Gather_3pt_4pt_Kernel.cu').read_text(),
                        'Prolongation_2D_Gather_3pt_4pt_Kernel')
prolongation_2d_3pt_4pt_gather_kernel.compile()

prolongation_2d_4pt_3pt_gather_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_2D_Gather_4pt_3pt_Kernel.cu').read_text(),
                        'Prolongation_2D_Gather_4pt_3pt_Kernel')
prolongation_2d_4pt_3pt_gather_kernel.compile()

prolongation_2d_4pt_gather_kernel = RawKernel(Path('Multigrid/Kernels/Prolongation_2D_Gather_4pt_Kernel.cu').read_text(),
                        'Prolongation_2D_Gather_4pt_Kernel')
prolongation_2d_4pt_gather_kernel.compile()



def prolongation(x: ndarray,x_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    '''
    Compute the 2D prolongation of the solution of the error equation from the coarse grid to the fine grid.
    The prolongation operation is the transpose of restriction.

    :param x: ndarray
    :param x_coarse: ndarray
    :param Nx: int
    :param Ny: int
    :param Nx_2h: int
    :param Ny_2h: int
    :return: x : ndarray, Updated solution
    '''

    if (Nx % 2 == 0) and (Ny % 2 == 0):
        prolongation_2d_gather_4pt(x, x_coarse, Nx, Ny, Nx_2h, Ny_2h)

    if (Nx % 2 != 0) and (Ny % 2 == 0):
        prolongation_2d_gather_4pt_3pt(x, x_coarse, Nx, Ny, Nx_2h, Ny_2h)
    if (Nx % 2 == 0) and (Ny % 2 != 0):
        prolongation_2d_gather_3pt_4pt(x, x_coarse, Nx, Ny, Nx_2h, Ny_2h)

    if (Nx % 2 != 0) and (Ny % 2 != 0):
        prolongation_2d_gather_3pt(x, x_coarse, Nx, Ny, Nx_2h, Ny_2h)

    return


def prolongation_2d_gather_4pt(x: ndarray, x_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    prolongation_2d_4pt_gather_kernel(*prolongation_2d_gather_kernel_config(Nx, Ny),
                          (x, x_coarse, Nx_2h, Ny_2h, Nx, Ny))

    return

def prolongation_2d_gather_3pt(x: ndarray, x_coarse: ndarray,Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    prolongation_2d_3pt_gather_kernel(*prolongation_2d_gather_kernel_config(Nx, Ny),
                          (x, x_coarse, Nx_2h, Ny_2h, Nx, Ny))

    return

def prolongation_2d_gather_4pt_3pt(x: ndarray, x_coarse: ndarray,Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    prolongation_2d_4pt_3pt_gather_kernel(*prolongation_2d_gather_kernel_config(Nx, Ny),
                          (x, x_coarse, Nx_2h, Ny_2h, Nx, Ny))

    return

def prolongation_2d_gather_3pt_4pt(x: ndarray, x_coarse: ndarray,Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    prolongation_2d_3pt_4pt_gather_kernel(*prolongation_2d_gather_kernel_config(Nx, Ny),
                          (x, x_coarse, Nx_2h, Ny_2h, Nx, Ny))

    return





def prolongation_2d_3pt(x: ndarray, x_coarse: ndarray,Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    prolongation_2d_3pt_kernel(*prolongation_2d_kernel_config(Nx_2h, Ny_2h),
                          (x, x_coarse, Nx_2h, Ny_2h, Nx, Ny))

    return


def prolongation_2d_4pt(x: ndarray, x_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    prolongation_2d_4pt_kernel(*prolongation_2d_kernel_config(Nx_2h, Ny_2h),
                          (x, x_coarse, Nx_2h, Ny_2h, Nx, Ny))

    return

def prolongation_2d_3pt_4pt(x: ndarray, x_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    prolongation_2d_3pt_4pt_kernel(*prolongation_2d_kernel_config(Nx_2h, Ny_2h),
                               (x, x_coarse, Nx_2h, Ny_2h, Nx, Ny))

    return

def prolongation_2d_4pt_3pt(x: ndarray, x_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    prolongation_2d_4pt_3pt_kernel(*prolongation_2d_kernel_config(Nx_2h, Ny_2h),
                               (x, x_coarse, Nx_2h, Ny_2h, Nx, Ny))

    return

def prolongation_2d_gather_kernel_config(Nx: int,Ny: int):

    threads_y = 16
    threads_x = 16

    block_x = (Ny + threads_y - 1)//threads_y
    block_y = (Nx + threads_x - 1)//threads_x

    return (block_x, block_y,), (threads_y,threads_x,)

def prolongation_2d_kernel_config(Nx: int,Ny: int):

    threads = 16

    block_x = (Nx + threads - 1)//threads
    block_y = (Ny + threads - 1)//threads

    return (block_x, block_y,), (threads,threads,)


def prolongation_warp_config(SIZE):
    threads = 256
    warps_per_block = threads // 32
    blocks = (SIZE + warps_per_block - 1)//warps_per_block
    return (blocks,),(threads,)

def prolongation_threads_config(SIZE):
    threads = 256
    blocks = (SIZE + threads - 1)//threads
    return (blocks,),(threads,)


# def compute_prolongation(A,x_coarse,x):
#     '''
#     Fused kernel computing the prolongation x = x + I @ x_coarse,
#     switching between threads-based and warp-based kernels
#     depending upon the average number of non-zero elements per row of A.
#
#     :param A: csr_matrix, prolongation operator
#     :param x_coarse: coarse grid solution vector
#     :param x: fine grid solution vector
#     :return: x: fine grid solution vector
#     '''
#     average_sparsity = A.nnz // A.shape[0]
#
#     if average_sparsity < 15:
#         prolongation_threads_kernel(*prolongation_threads_config(A.shape[0]),
#                            (A.shape[0],
#                                 A.data,
#                                 A.indptr,
#                                 A.indices,
#                                 x_coarse,x))
#     else:
#         prolongation_warp_kernel(*prolongation_warp_config(A.shape[0]),
#                        (A.shape[0],
#                             A.data,
#                             A.indptr,
#                             A.indices,
#                             x_coarse, x))
#
#     return x