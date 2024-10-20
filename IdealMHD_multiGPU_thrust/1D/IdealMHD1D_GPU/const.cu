#include "const.hpp"


void initializeDeviceConstants() {
    cudaMemcpyToSymbol(device_EPS, &EPS, sizeof(double));
    cudaMemcpyToSymbol(device_PI, &PI, sizeof(double));
    cudaMemcpyToSymbol(device_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(device_xmin, &xmin, sizeof(double));
    cudaMemcpyToSymbol(device_xmax, &xmax, sizeof(double));
    cudaMemcpyToSymbol(device_nx, &nx, sizeof(int));
    cudaMemcpyToSymbol(device_CFL, &CFL, sizeof(double));
    cudaMemcpyToSymbol(device_gamma_mhd, &gamma_mhd, sizeof(double));
    cudaMemcpyToSymbol(device_dt, &dt, sizeof(double));
    cudaMemcpyToSymbol(device_totalStep, &totalStep, sizeof(int));
    cudaMemcpyToSymbol(totalTime, &device_totalTime, sizeof(double));
}
