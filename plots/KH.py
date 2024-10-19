import numpy as np 
import matplotlib.pyplot as plt 
import os 


gamma = 5.0/3.0
B0 = 1.0
rho0 = 1.0
shear_thickness = 1.0
beta = 2.0
p0 = beta * B0**2 / 2.0
rr = 0.2
br = 1.0
theta = np.pi / 2.0
V0 = Vf = np.sqrt(B0**2 / rho0 + gamma * p0 / rho0)
x_max = 2.0 * np.pi * shear_thickness / 0.4
y_max = 10.0 * shear_thickness
dx = shear_thickness / 32.0
dy = dx
nx = int(x_max / dx)
ny = int(2.0 * y_max / dy)
dt = 0.0
CFL = 0.7
x = np.arange(0.0, x_max - dx, dx)
y = np.arange(-y_max, y_max, dy)
X, Y = np.meshgrid(x, y)


procs = 4

dirname = f"/cfca-work/akutagawakt/IdealMHD_multiGPU/results_KH_procs={procs}"
log = np.loadtxt(f"{dirname}/log_KH.txt", delimiter=',')
total_steps = int(log[-1][0])
interval = 100

step = 12000
savename = f"{step}_{procs}.png"

grid_x = int(log[0][0])
grid_y = int(log[0][1])

fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)

total_rho = np.zeros([nx, ny])
for rank in range(procs):
    local_grid_x = rank // grid_x
    local_grid_y = rank % grid_x

    filename = f"{dirname}/KH_{step}_{rank}.bin"
    with open(filename, 'rb') as f:
        U = np.fromfile(f, dtype=np.float64)
    U = U.reshape(int(nx / grid_x), int(ny / grid_y), 8).T

    rho = U[0, :]
    u = U[1, :] / rho 
    v = U[2, :] / rho 
    w = U[3, :] / rho 
    Bx = U[4, :]
    By = U[5, :]
    Bz = U[6, :]
    e = U[7, :]
    p = (gamma-1) * (e - rho*(u**2+v**2+w**2)/2 - (Bx**2+By**2+Bz**2)/2)
    magnetic_pressure = 1/2 * (Bx**2+By**2+Bz**2)
    pT = p + magnetic_pressure

    total_rho[
        int(local_grid_x * nx / grid_x) : int((local_grid_x + 1) * nx / grid_x), 
        int(local_grid_y * ny / grid_y) : int((local_grid_y + 1) * ny / grid_y)
    ] = rho.T


contour = ax1.pcolormesh(X, Y, total_rho.T, cmap='jet')
cbar = plt.colorbar(contour, ax=ax1)

ax1.set_title(r"$\rho$", fontsize=24)
#ax1.text(0.8, 1.02, f"t = {log[int(step/100)][1]:.3f}", transform=ax1.transAxes, fontsize=16)
ax1.tick_params(labelsize=16)

fig.savefig(savename, dpi=200)


