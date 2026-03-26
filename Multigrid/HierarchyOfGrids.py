from Multigrid.Grid import Grid
from Multigrid.NumberOfGrids import number_of_grids
import numpy as np
from scipy.sparse import csr_matrix,kron
from SparseApproximateInverse.SPAI_0 import spai_0
from numpy import float32,zeros
from numba import njit, prange

def hierarchy_of_grids(matrix,b,x0,Nx: int,Ny: int):
    '''
        Create the heirarchy of restricted grids for multigrid on the CPU.
    :param matrix: csr_matrix, fine grid matrix
    :param b: ndarray, initial b vector
    :param x0: ndarray, initial x vector
    :param Nx: int
    :param Ny: int
    :return: list[Grid] : list of Grid objects
    '''
    numgrids: int = number_of_grids(np.min([Nx,Ny]))

    # initialise
    grids = [Grid() for i in range(numgrids)]

    # fine grid
    r_2d: csr_matrix
    i_2d: csr_matrix
    nx_2h: int
    ny_2h: int

    r_2d,i_2d, nx_2h, ny_2h = restriction_prolongation_operator(Nx, Ny)
    M = spai_0(matrix)

    grids[0] = Grid(matrix,M,x0,b,r_2d,i_2d,np.zeros_like(b,dtype=float32),Nx,Ny)

    grids[1].Nx = int(nx_2h)
    grids[1].Ny = int(ny_2h)



    for i in range(1,numgrids):

        grids[i].Matrix = grids[i - 1].R @ grids[i - 1].Matrix @ grids[i - 1].I
        grids[i].Matrix.sort_indices()
        grids[i].M = spai_0(grids[i].Matrix)
        grids[i].M.sort_indices()
        grids[i].x = np.zeros((grids[i].Nx * grids[i].Ny), dtype=float32)
        grids[i].temp_array = np.zeros((grids[i].Nx * grids[i].Ny), dtype=float32)

        if i < (numgrids - 1):
            r_2d, i_2d, nx_2h, ny_2h = restriction_prolongation_operator(grids[i].Nx, grids[i].Ny)

            grids[i].R = r_2d
            grids[i].R.sort_indices()
            grids[i].I = i_2d
            grids[i].I.sort_indices()
            grids[i].b = np.zeros((grids[i].Nx * grids[i].Ny), dtype=float32)
            grids[i+1].Nx = nx_2h
            grids[i+1].Ny = ny_2h

        if i == numgrids -1:
            grids[i].b = np.zeros((grids[i].Nx * grids[i].Ny), dtype=float32)

    return grids,numgrids



def restriction_prolongation_operator(Nx,Ny):

    # odd even kernels
    #linear weighting
    k_n_odd = np.array([1.0,2.0,1.0], dtype=float32)
    #spline weighting
    k_n_even = np.array([1.0,3.0,3.0,1.0], dtype=float32)

    # reduced dimensions
    Nx_2h = (Nx + 1)//2 - 1
    Ny_2h = (Ny + 1)//2 - 1

    #Choose stencil based upon whether even or odd N
    if 0 == np.mod(Nx, 2):
        stencil_x =  1/8 * k_n_even
    else:
        stencil_x =  1/4 * k_n_odd

    if 0 == np.mod(Ny, 2):
        stencil_y = 1/8 * k_n_even
    else:
        stencil_y = 1/4 * k_n_odd


    #Make restriction matrices
    Rx_2h = csr_matrix(restriction_operator_csr_1d(Nx_2h, stencil_x),
                       shape=(Nx_2h,Nx), dtype=np.float32)
    Ry_2h = csr_matrix(restriction_operator_csr_1d(Ny_2h, stencil_y),
                       shape=(Ny_2h, Ny), dtype=np.float32)

    #2D restriction is Kronecker product kron(R_2h,R_2h)
    r_2d: csr_matrix = kron(Rx_2h, Ry_2h,format='csr')

    # Prolongation operator is transpose of restriction operator
    Ix_2h = csr_matrix(Rx_2h.transpose(), shape=(Nx, Nx_2h), dtype=np.float32)
    Iy_2h = csr_matrix(Ry_2h.transpose(), shape=(Ny, Ny_2h), dtype=np.float32)
    i_2d = float32(4.0) * csr_matrix(kron(Ix_2h, Iy_2h, format='csr'))

    return r_2d,i_2d,Nx_2h,Ny_2h



@njit(parallel=True)
def restriction_operator_csr_1d(n2h,stencil):
    stencil_length = len(stencil)

    rowptr = zeros((n2h + 1),dtype=np.int32)
    col = zeros((stencil_length * n2h),dtype=np.int32)
    val = zeros((stencil_length * n2h),dtype=float32)

    for i in prange(n2h):

        for k in range(stencil_length):
            col[i*stencil_length +k] = 2*i + k
            val[i*stencil_length +k] = stencil[k]

        rowptr[i+1] = i*stencil_length + stencil_length

    return (val, col, rowptr)


