#ifndef CONST_STRUCT_H
#define CONST_STRUCT_H

extern const double EPS;
extern const double PI;

extern const double dx;
extern const double xmin;
extern const double xmax;
extern const int nx;

extern const double CFL;
extern const double gamma_mhd;

extern double dt;

extern const int totalStep;
extern double totalTime;


extern __constant__ double device_EPS;
extern __constant__ double device_PI;

extern __constant__ double device_dx;
extern __constant__ double device_xmin;
extern __constant__ double device_xmax;
extern __constant__ int device_nx;

extern __constant__ double device_CFL;
extern __constant__ double device_gamma_mhd;

extern __device__ double device_dt;

extern __constant__ int device_totalStep;
extern __device__ double device_totalTime;


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

#endif


