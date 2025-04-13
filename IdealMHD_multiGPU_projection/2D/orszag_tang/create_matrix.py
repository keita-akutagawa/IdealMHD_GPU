import numpy as np
import scipy.sparse as sp
import scipy.io as sio


def make_poisson_matrix_periodic(Nx, Ny, dx, dy):
    Ix = sp.eye(Nx, format='csr')
    Iy = sp.eye(Ny, format='csr')

    ex = np.ones(N) / (2.0 * dx)**2
    Tx = sp.diags([-ex, 2*ex, -ex], [-2, 0, 2], shape=(Nx, Nx), format='lil')
    Tx[0, -2] = -1 / (2.0 * dx)**2 
    Tx[1, -1] = -1 / (2.0 * dx)**2 
    Tx[-2, 0] = -1 / (2.0 * dx)**2 
    Tx[-1, 1] = -1 / (2.0 * dx)**2 

    ey = np.ones(N) / (2.0 * dy)**2
    Ty = sp.diags([-ey, 2*ey, -ey], [-2, 0, 2], shape=(Ny, Ny), format='lil')
    Ty[0, -2] = -1 / (2.0 * dy)**2 
    Ty[1, -1] = -1 / (2.0 * dy)**2 
    Ty[-2, 0] = -1 / (2.0 * dy)**2 
    Ty[-1, 1] = -1 / (2.0 * dy)**2 

    A = sp.kron(Iy, Tx) + sp.kron(Ty, Ix)
    
    return A


Nx, Ny = 256, 256
dx, dy = 2.0 * np.pi / Nx, 2.0 * np.pi / Ny
N = Nx * Ny

A = make_poisson_matrix_periodic(Nx, Ny, dx, dy)

sio.mmwrite("poisson_periodic.mtx", A)


