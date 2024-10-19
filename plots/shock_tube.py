import numpy as np 
import matplotlib.pyplot as plt 
import os 


gamma = 5.0/3.0
B0 = 1.0
rho0 = 1.0
p0 = 1.0
x_max = 1.0
dx = 0.001
nx = int(x_max / dx)
dt = 0.0
CFL = 0.7
x_coordinate = np.arange(0.0, x_max, dx)


testname = "5b"
dirname = "/cfca-work/akutagawakt/IdealMHD_multiGPU/results_shock_tube_test" + testname
#log = np.loadtxt(f"{dirname}/log_shock_tube.txt", delimiter=',')
#total_steps = int(log[-1][0])
interval = 100

step = 700
savename = f"{testname}_{step}.png"

procs = 4


fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

total_rho = np.zeros(nx)
total_p = np.zeros(nx)
total_u = np.zeros(nx)
total_v = np.zeros(nx)
total_By = np.zeros(nx)
total_Bz = np.zeros(nx)
for rank in range(procs):

    filename = f"{dirname}/shock_tube_{step}_{rank}.bin"
    with open(filename, 'rb') as f:
        U = np.fromfile(f, dtype=np.float64)
    U = U.reshape(-1, 8).T

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

    total_rho[int(rank * nx / procs) : int((rank + 1) * nx / procs)] = rho
    total_p  [int(rank * nx / procs) : int((rank + 1) * nx / procs)] = p
    total_u  [int(rank * nx / procs) : int((rank + 1) * nx / procs)] = u
    total_v  [int(rank * nx / procs) : int((rank + 1) * nx / procs)] = v
    total_By [int(rank * nx / procs) : int((rank + 1) * nx / procs)] = By
    total_Bz [int(rank * nx / procs) : int((rank + 1) * nx / procs)] = Bz


ax1.plot(x_coordinate, total_rho, marker='o', markersize=3)
ax2.plot(x_coordinate, total_p,   marker='o', markersize=3)
ax3.plot(x_coordinate, total_u,   marker='o', markersize=3)
ax4.plot(x_coordinate, total_v,   marker='o', markersize=3)
ax5.plot(x_coordinate, total_By,  marker='o', markersize=3)
ax6.plot(x_coordinate, total_Bz,  marker='o', markersize=3)
ax1.set_title(r"$\rho$", fontsize=24)
ax2.set_title(r"$p$", fontsize=24)
ax3.set_title(r"$u$", fontsize=24)
ax4.set_title(r"$v$", fontsize=24)
ax5.set_title(r"$B_y$", fontsize=24)
ax6.set_title(r"$B_z$", fontsize=24)
ax1.tick_params(labelsize=16)
ax2.tick_params(labelsize=16)
ax3.tick_params(labelsize=16)
ax4.tick_params(labelsize=16)
ax5.tick_params(labelsize=16)
ax6.tick_params(labelsize=16)
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()

fig.savefig(savename, dpi=200)


