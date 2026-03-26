from Multigrid.Grid import Grid
from Multigrid.Grid_gpu import Grid_GPU
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from cupy import asarray
from numpy import float32

def grids_to_gpu(grids:list[Grid]):
    '''
        Allocate GPU arrays and csr_matrices from existing CPU grids.
    :param grids: list of Grid objects
    :return: list of GPU grid objects
    '''
    # initialise
    grids_gpu = [Grid_GPU() for i in range(len(grids))]

    for n in range(len(grids)):
        grids_gpu[n].Matrix = csr_matrix_gpu(grids[n].Matrix,dtype=float32)
        grids_gpu[n].Matrix.sort_indices()
        grids_gpu[n].M = csr_matrix_gpu(grids[n].M, dtype=float32)
        grids_gpu[n].M.sort_indices()
        grids_gpu[n].R = csr_matrix_gpu(grids[n].R, dtype=float32)
        grids_gpu[n].R.sort_indices()
        grids_gpu[n].I = csr_matrix_gpu(grids[n].I, dtype=float32)
        grids_gpu[n].I.sort_indices()
        grids_gpu[n].x = asarray(grids[n].x, dtype=float32)
        grids_gpu[n].b = asarray(grids[n].b, dtype=float32)
        grids_gpu[n].temp_array = asarray(grids[n].temp_array, dtype=float32)
        grids_gpu[n].Nx = grids[n].Nx
        grids_gpu[n].Ny = grids[n].Ny

    return grids_gpu