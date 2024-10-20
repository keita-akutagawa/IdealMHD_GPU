#include "../const.hpp"

const double EPS = 1e-40;
const double PI = 3.141592653589793;

const double dx = 0.01;
const double xmin = 0.0;
const double xmax = 1.0;
const int nx = static_cast<int>((xmax - xmin) / dx);

const double CFL = 0.7;
const double gamma_mhd = 5.0 / 3.0;

double dt = 0.0;

const int totalStep = 100;
double totalTime = 0.0;


__constant__ double device_EPS;
__constant__ double device_PI;

__constant__ double device_dx;
__constant__ double device_xmin;
__constant__ double device_xmax;
__constant__ int device_nx;

__constant__ double device_CFL;
__constant__ double device_gamma_mhd;

__device__ double device_dt;

__constant__ int device_totalStep;
__device__ double device_totalTime;


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
