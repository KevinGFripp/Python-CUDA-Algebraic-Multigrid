
def number_of_grids(N: int):
    '''
    :param N: int
    :return: number of grids
    '''
    numGRIDS: int = 1
    N_new: int = N

    while N_new > 4:
        N_new = (N_new + 1)//2 - 1
        numGRIDS += 1

    return numGRIDS
