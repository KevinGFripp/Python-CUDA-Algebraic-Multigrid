from scipy.sparse import diags_array
from numpy import float32

def laplacian_1d(n):
    offsets = [-1, 0, 1]
    diagonals = [1.0,-2.0,1.0]

    return diags_array(diagonals, offsets=offsets, shape=(n, n), format='csr',dtype=float32)
