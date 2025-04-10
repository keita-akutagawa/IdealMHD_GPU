import numpy as np
import scipy.sparse as sp
import scipy.io as sio


def make_poisson_matrix_periodic(Nx, Ny, dx, dy):
    Ix = sp.eye(Nx, format='csr')
    Iy = sp.eye(Ny, format='csr')

    ex = np.ones(Nx) / dx**2
    Tx = sp.diags([-ex, 2*ex, -ex], [-1, 0, 1], shape=(Nx, Nx), format='lil')
    Tx[0, -1] = 1
    Tx[-1, 0] = 1
    Tx = Tx.tocsr()

    ey = np.ones(Ny) / dy**2
    Ty = sp.diags([-ey, 2*ey, -ey], [-1, 0, 1], shape=(Ny, Ny), format='lil')
    Ty[0, -1] = 1
    Ty[-1, 0] = 1
    Ty = Ty.tocsr()

    A = sp.kron(Iy, Tx) + sp.kron(Ty, Ix)
    return A


Nx, Ny = 256, 256
dx, dy = 2.0 * np.pi / Nx, 2.0 * np.pi / Ny 
N = Nx * Ny

A = make_poisson_matrix_periodic(Nx, Ny, dx, dy)

sio.mmwrite("poisson_periodic.mtx", A)

