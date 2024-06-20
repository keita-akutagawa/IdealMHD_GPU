#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../lib_IdealMHD_1D_GPU/const.hpp"
#include "../../lib_IdealMHD_1D_GPU/idealMHD_1D.hpp"

const double EPS = 1e-20;
const double PI = 3.141592653589793;
const double dx = 0.001;
const double xmin = 0.0;
const double xmax = 1.0;
const int nx = int((xmax - xmin) / dx);
const double CFL = 0.7;
const double gamma_mhd = 5.0 / 3.0;
double dt = 0.0;
const int totalStep = 1000;
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


__global__ void initializeU_kernel(
    ConservationParameter* U, 
    double rho0, double u0, double v0, double w0, double bX0, double bY0, double bZ0, double p0, double e0
) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < device_nx) {
        U[i].rho  = rho0;
        U[i].rhoU = rho0 * u0;
        U[i].rhoV = rho0 * v0;
        U[i].rhoW = rho0 * w0;
        U[i].bX   = bX0;
        U[i].bY   = bY0;
        U[i].bZ   = bZ0;
        U[i].e    = e0;
    }
}

void IdealMHD1D::initializeU()
{
    double rho0, u0, v0, w0, bX0, bY0, bZ0, p0, e0;

    rho0 = 1.0;
    u0 = 1.0; v0 = 0.0; w0 = 0.0;
    bX0 = 1.0; bY0 = 0.0; bZ0 = 0.0;
    p0 = 1.0;
    e0 = p0 / (gamma_mhd - 1.0)
        + 0.5 * rho0 * (u0 * u0 + v0 * v0 + w0 * w0)
        + 0.5 * (bX0 * bX0 + bY0 * bY0 + bZ0 * bZ0);

    

    int threadsPerBlock = 256;
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        rho0, u0, v0, w0, bX0, bY0, bZ0, p0, e0
    );

    cudaDeviceSynchronize();
}


int main()
{
    initializeDeviceConstants();

    std::string directoryname = "results";
    std::string filenameWithoutStep = "wave";
    std::ofstream logfile("log.txt");
    int recordStep = 100;

    IdealMHD1D idealMHD1D;

    idealMHD1D.initializeU();
    std::cout << "AAA";

    for (int step = 0; step < totalStep+1; step++) {
        if (step % recordStep == 0) {
            idealMHD1D.save(directoryname, filenameWithoutStep, step);
            logfile << std::to_string(step) << ","
                    << std::setprecision(3) << totalTime
                    << std::endl;
        }
        std::cout << "AAA";
        idealMHD1D.oneStepRK2();
        totalTime += dt;
    }
    
    return 0;
}


