from Multigrid.AMG_GPU import amg_cycle_preconditioner as amg
from Multigrid.Grid_gpu import Grid_GPU as Grid
from cupy.random import rand
from cupy import ndarray, zeros_like, dot,asarray
from cupy.linalg import norm
from numpy import float32,max
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
    norm_b = norm(b).get()
    r_norm = norm(r).get()


    iterations = 0.0

    for iteration in range(max_iterations):
        # apply left preconditioning
        y = amg(grids, P, amg_x0, cycle=cycle)

        v: ndarray(dtype=float32) = grids[0].Matrix @ y
        alpha = rho / dot(rhat, v)
        h = ax_plus_by(1.,x,alpha,y)
        s = ax_plus_by(1.,r,-alpha,v)

        r_norm = norm(s).get()

        iterations += 0.5
        if converged(r_norm,norm_b,float32(tol),float32(1e-6)):
            x = h
            break

        # apply left preconditioning
        z = amg(grids, s, amg_x0, cycle=cycle)

        t = grids[0].Matrix @ z
        omega = omega_dot_ratio(s,t)
        x = ax_plus_by(1.,h,omega,z)
        r = ax_plus_by(1.,s,-omega,t)

        #r_norm = norm(r) * inv_norm_b
        r_norm = norm(r).get()

        iterations += 0.5
        if converged(r_norm,norm_b,float32(tol),float32(1e-6)):
            break

        rho_1 = dot(rhat,r)
        beta = beta_ratio(rho_1,rho,alpha,omega)
        P = ax_plus_by(1.,r,beta,ax_plus_by(1.,P,-omega,v))

        rho = rho_1

    return x, r_norm/norm_b, iterations

def converged(r: float32, b: float32, reltol: float32, abstol: float32) -> bool:

    return r <= max([b*reltol,abstol])




