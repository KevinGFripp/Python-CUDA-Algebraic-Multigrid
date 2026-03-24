from scipy.sparse import csr_matrix,diags
from numpy import ndarray,add,zeros_like,float32


def spai_0(A: csr_matrix):
    '''
        Sparse approximate inverse order "0", row-weighted diagonal Jacobi method
        M_ii = A_ii / || A_i||^2
    :param A: csr_matrix
    :return: M: csr_matrix, diagonal spai_0
    '''


    diagonal = A.diagonal()
    data = A.data
    indptr = A.indptr

    # perform the row reduction
    row_norm_squared = add.reduceat(data*data,indptr[:-1])
    spai_diagonal = zeros_like(diagonal)

    #catch divide by 0s
    mask = row_norm_squared > 0
    spai_diagonal[mask] = diagonal[mask]/row_norm_squared[mask]

    return csr_matrix(diags(spai_diagonal,format='csr',dtype=float32))

def spai(A: csr_matrix,M: csr_matrix,b: ndarray,x: ndarray,iterations: int):
    '''
        Perform sparse approximate inverse order 0 iterations with the preconditioner matrix M.
        :param A: csr_matrix
        :param M: csr_matrix, diagonal spai_0
        :param b: ndarray
        :param x: ndarray
        :param iterations: int
        :return: x: solution
        '''
    for n in range(iterations):
        x = x + M @ (b - A @ x)

    return x

def spai_x0_0(A: csr_matrix,M: csr_matrix,b: ndarray,x: ndarray,iterations: int):
    '''
        Optimised iterations for special case of x0 =0, which reduces to M @ b for the first iteration.
    :param A: csr_matrix
    :param M: csr_matrix, diagonal spai_0
    :param b: ndarray
    :param x: ndarray
    :param iterations: int
    :return: x: solution
    '''
    x = M @ b
    for n in range(1,iterations):
        x += M @ (b - A @ x)

    return x