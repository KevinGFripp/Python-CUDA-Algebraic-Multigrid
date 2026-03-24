from Multigrid.AMG import amg_cycle_preconditioner as amg
from Multigrid.Grid import Grid
from numpy.random import rand
from numpy import ndarray, zeros_like, dot
from numpy.linalg import norm
from numpy import float32, astype


def amg_bicgstab_l(grids: list[Grid], b: ndarray, x0: ndarray, max_iterations: int, tol: float, cycle):
    '''
    CPU-based AMG left-preconditioned bicgstab(1) iterative solver.

    :param grids: list of Grid objects for AMG
    :param b: numpy ndarray
    :param x0: numpy ndarray
    :param max_iterations: int
    :param tol: float
    :param cycle: Multigrid cycle, 'V' or 'F'
    :return: x: solution ,r_norm: last iteration residual, iterations: total number of iterations
    '''

    # random initial starting residual
    rhat: ndarray = rand(grids[0].Nx*grids[0].Ny)
    rhat = astype(rhat, float32)

    x = x0
    r = b - grids[0].Matrix @ x
    rho = dot(rhat,r)
    P: ndarray = r
    norm_b = norm(b) if norm(b) != 0 else 1.0
    r_norm = norm(r)/norm_b

    #amg x0
    amg_x0 = zeros_like(x0, dtype=float32)

    iterations = 0.0
    for iteration in range(max_iterations):
        # apply left preconditioning
        y = amg(grids, P, amg_x0, cycle=cycle)

        v = grids[0].Matrix @ y
        alpha = rho / dot(rhat,v)
        h = x + alpha * y
        s  = r - alpha * v

        r_norm = norm(s) / norm_b

        iterations += 0.5
        if r_norm < tol:
            x = h
            break

        # apply left preconditioning
        z = amg(grids, s, amg_x0, cycle=cycle)

        t = grids[0].Matrix @ z
        omega = dot(t, s) / dot(t, t)
        x = h + omega * z
        r = s - omega * t

        r_norm = norm(r)/norm_b

        iterations += 0.5
        if r_norm < tol:
            break

        rho_1 = dot(rhat,r)
        beta = (rho_1 / rho) * (alpha / omega)
        P = r + beta * (P - omega * v)
        rho = rho_1

    return x, r_norm, iterations
