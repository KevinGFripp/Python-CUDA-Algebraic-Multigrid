from scipy.sparse import identity, kron, csr_matrix
from Laplacians.Laplacian1D import laplacian_1d
from numpy import float32

def laplacian_2d(nx,ny):
    laplacian_x = laplacian_1d(nx)
    laplacian_y = laplacian_1d(ny)

    identity_x = identity(nx, dtype=float32)
    identity_y = identity(ny, dtype=float32)

    del2 = kron(laplacian_x, identity_y) + kron(identity_x,laplacian_y)

    return csr_matrix(del2)