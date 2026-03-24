from cupy import RawKernel
from cupy import ndarray
from pathlib import Path


residual_restriction_2d_3pt_kernel = RawKernel(
                                     Path('Multigrid/Kernels/Restriction_NyNx_3pt_Fused_Kernel.cu').read_text(),
                                  'restriction_2d_convolution_3pt_fused')
residual_restriction_2d_3pt_kernel.compile()

residual_restriction_2d_4pt_kernel = RawKernel(
                                     Path('Multigrid/Kernels/Restriction_NyNx_4pt_Fused_Kernel.cu').read_text(),
                                  'restriction_2d_convolution_4pt_fused')
residual_restriction_2d_4pt_kernel.compile()

residual_restriction_2d_3pt_4pt_kernel = RawKernel(
                                     Path('Multigrid/Kernels/Restriction_NyNx_3pt_4pt_Fused_Kernel.cu').read_text(),
                                  'restriction_2d_convolution_3pt_4pt_fused')
residual_restriction_2d_3pt_4pt_kernel.compile()

residual_restriction_2d_4pt_3pt_kernel = RawKernel(
                                     Path('Multigrid/Kernels/Restriction_NyNx_4pt_3pt_Fused_Kernel.cu').read_text(),
                                  'restriction_2d_convolution_4pt_3pt_fused')
residual_restriction_2d_4pt_3pt_kernel.compile()


def restriction(r: ndarray,b: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):
    '''
    Compute the 2D restriction of the residual from the fine grid to the coarse grid,
    using a fused shared memory 2D stencil kernel.
    :param r: ndarray
    :param b: ndarray
    :param Nx: int
    :param Ny: int
    :param Nx_2h: int
    :param Ny_2h: int
    :return: b : ndarray, right-hand side on the coarse grid error equation, holding the restricted
                          fine grid residual.
    '''

    if (Nx % 2 == 0) and (Ny % 2 == 0):
        b = restriction_ny_nx_4pt(r,b,Nx,Ny,Nx_2h,Ny_2h)

    if (Nx % 2 != 0) and (Ny % 2 == 0):
        b = restriction_ny_nx_4pt_3pt(r,b,Nx,Ny,Nx_2h,Ny_2h)

    if (Nx % 2 == 0) and (Ny % 2 != 0):
        b = restriction_ny_nx_3pt_4pt(r,b,Nx,Ny,Nx_2h,Ny_2h)

    if (Nx % 2 != 0) and (Ny % 2 != 0):
        b = restriction_ny_nx_3pt(r,b,Nx,Ny,Nx_2h,Ny_2h)

    return b


def restriction_ny_nx_4pt(r: ndarray,r_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):

    residual_restriction_2d_4pt_kernel(*kernel_tiled_2d_config(Nx_2h, Ny_2h),
                                                 (r,r_coarse,
                                                       Nx, Ny,
                                                       Nx_2h, Ny_2h),
                                                 shared_mem=kernel_tiled_2d_4pt_shared_size())
    return r_coarse

def restriction_ny_nx_3pt(r: ndarray,r_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):

    residual_restriction_2d_3pt_kernel(*kernel_tiled_2d_config(Nx_2h, Ny_2h),
                                                 (r,r_coarse,
                                                       Nx, Ny,
                                                       Nx_2h, Ny_2h),
                                                 shared_mem=kernel_tiled_2d_3pt_shared_size())
    return r_coarse

def restriction_ny_nx_3pt_4pt(r: ndarray,r_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):

    residual_restriction_2d_3pt_4pt_kernel(*kernel_tiled_2d_config(Nx_2h, Ny_2h),
                                                 (r,r_coarse,
                                                       Nx, Ny,
                                                       Nx_2h, Ny_2h),
                                                 shared_mem=kernel_tiled_2d_3pt_4pt_shared_size())
    return r_coarse

def restriction_ny_nx_4pt_3pt(r: ndarray,r_coarse: ndarray, Nx: int, Ny: int, Nx_2h: int, Ny_2h: int):

    residual_restriction_2d_4pt_3pt_kernel(*kernel_tiled_2d_config(Nx_2h, Ny_2h),
                                                 (r,r_coarse,
                                                       Nx, Ny,
                                                       Nx_2h, Ny_2h),
                                                 shared_mem=kernel_tiled_2d_3pt_4pt_shared_size())
    return r_coarse



def kernel_tiled_2d_config(Nx: int, Ny: int):

    threads = 16

    block_x = (Nx + threads - 1) // threads
    block_y = (Ny + threads - 1) // threads

    return (block_x, block_y,), (threads,threads,)


def kernel_tiled_2d_3pt_4pt_shared_size():
    threads = 16

    return (2 * threads + 3) * (2 * threads + 2) * 4

def kernel_tiled_2d_3pt_shared_size():
    threads = 16

    return (2 * threads + 2) * (2 * threads + 2) * 4

def kernel_tiled_2d_4pt_shared_size():
    threads = 16

    return (2 * threads + 3) * (2 * threads + 3) * 4

def kernel_warp_config(Nx: int,Ny: int):
    warps = Nx
    threads = 128

    grid = (warps * 32 + threads - 1)//threads

    return (grid,),(threads,)

def kernel_tiled_4pt_shared_size(Nx, Ny):
    threads = 16

    size_of_float = 4

    return size_of_float * (2 * threads + 3) * threads

def kernel_tiled_3pt_shared_size(Nx, Ny):
    threads = 16

    size_of_float = 4

    return size_of_float * (2 * threads + 2) * threads


def kernel_tiled_config(Nx: int,Ny: int):

    threads = 16

    block_x = (Nx + threads - 1)// threads
    block_y = (Ny + threads - 1)// threads

    return (block_x,block_y,),(threads,threads,)

def kernel_config(Nx: int,Ny: int):

    threads = 256
    block_x = (Nx*Ny + threads - 1) // threads

    return (block_x,), (threads,)

def kernel_2d_config(Nx: int,Ny: int):

    threadx = 32
    thready = 8

    blockx = (Nx + threadx - 1)// threadx
    blocky = (Ny + thready - 1)// thready

    return (blockx,blocky,),(threadx,thready,)



