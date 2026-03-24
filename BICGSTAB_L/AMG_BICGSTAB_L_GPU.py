from Multigrid.AMG_GPU import amg_cycle_preconditioner as amg
from Multigrid.Grid_gpu import Grid_GPU as Grid
from cupy.random import rand
from cupy import ndarray, zeros_like, dot,asarray
from cupy.linalg import norm
from numpy import float32
from BICGSTAB_L.Kernels.BICGSTAB_kernels import omega_dot_ratio,beta_ratio,ax_plus_by


def amg_bicgstab_l_gpu(grids: list[Grid], b: ndarray, x0: ndarray, max_iterations: int, tol: float, cycle):
    '''
        GPU-based AMG left-preconditioned bicgstab(1) iterative solver.

        :param grids: list of GPU Grid objects for AMG
        :param b: cupy ndarray
        :param x0: cupy ndarray
        :param max_iterations: int
        :param tol: float
        :param cycle: Multigrid cycle, 'V' or 'F'
        :return: x: solution ,r_norm: last iteration residual, iterations: total number of iterations
        '''

    x = x0

    # random initial starting residual
    rhat = rand(grids[0].Nx*grids[0].Ny, dtype=float32)

    # amg x0 vector
    amg_x0 = zeros_like(x, dtype=float32)

    r: ndarray(dtype=float32) = b - grids[0].Matrix @ x
    rho: ndarray(dtype=float32) = rhat @ r
    P: ndarray(dtype=float32) = r
    inv_norm_b = asarray(1./norm(b),dtype=float32) if norm(b) != 0.0 else asarray(1.0,dtype=float32)
    r_norm = norm(r) * inv_norm_b


    iterations = 0.0

    for iteration in range(max_iterations):
        # apply left preconditioning
        y = amg(grids, P, amg_x0, cycle=cycle)

        v: ndarray(dtype=float32) = grids[0].Matrix @ y
        alpha = rho / dot(rhat, v)
        h = ax_plus_by(1.,x,alpha,y)
        s = ax_plus_by(1.,r,-alpha,v)

        r_norm: ndarray(dtype=float32) = norm(s) * inv_norm_b

        iterations += 0.5
        if r_norm.get() < tol:
            x = h
            break

        # apply left preconditioning
        z = amg(grids, s, amg_x0, cycle=cycle)

        t = grids[0].Matrix @ z
        omega = omega_dot_ratio(s,t)
        x = ax_plus_by(1.,h,omega,z)
        r = ax_plus_by(1.,s,-omega,t)

        r_norm = norm(r) * inv_norm_b

        iterations += 0.5
        if r_norm.get() < tol:
            break

        rho_1 = dot(rhat,r)
        beta = beta_ratio(rho_1,rho,alpha,omega)
        P = ax_plus_by(1.,r,beta,ax_plus_by(1.,P,-omega,v))

        rho = rho_1

    return x, r_norm, iterations



