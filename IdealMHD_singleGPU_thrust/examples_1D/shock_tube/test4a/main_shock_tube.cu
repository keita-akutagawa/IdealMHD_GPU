#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../../lib_IdealMHD_1D_GPU/const.hpp"
#include "../../../lib_IdealMHD_1D_GPU/idealMHD_1D.hpp"

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
    double rhoL0, double uL0, double vL0, double wL0, double bXL0, double bYL0, double bZL0, double pL0, double eL0, 
    double rhoR0, double uR0, double vR0, double wR0, double bXR0, double bYR0, double bZR0, double pR0, double eR0
) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < device_nx) {
        if (i < device_nx / 2) {
            U[i].rho  = rhoL0;
            U[i].rhoU = rhoL0 * uL0;
            U[i].rhoV = rhoL0 * vL0;
            U[i].rhoW = rhoL0 * wL0;
            U[i].bX   = bXL0;
            U[i].bY   = bYL0;
            U[i].bZ   = bZL0;
            U[i].e    = eL0;
        } else {
            U[i].rho  = rhoR0;
            U[i].rhoU = rhoR0 * uR0;
            U[i].rhoV = rhoR0 * vR0;
            U[i].rhoW = rhoR0 * wR0;
            U[i].bX   = bXR0;
            U[i].bY   = bYR0;
            U[i].bZ   = bZR0;
            U[i].e    = eR0;
        }
    }
}

void IdealMHD1D::initializeU()
{
    double rhoL0, uL0, vL0, wL0, bXL0, bYL0, bZL0, pL0, eL0;
    double rhoR0, uR0, vR0, wR0, bXR0, bYR0, bZR0, pR0, eR0;

    rhoL0 = 1.0;
    uL0 = 0.0; vL0 = 0.0; wL0 = 0.0;
    bXL0 = 1.0; bYL0 = 1.0; bZL0 = 0.0;
    pL0 = 1.0;
    eL0 = pL0 / (gamma_mhd - 1.0)
        + 0.5 * rhoL0 * (uL0 * uL0 + vL0 * vL0 + wL0 * wL0)
        + 0.5 * (bXL0 * bXL0 + bYL0 * bYL0 + bZL0 * bZL0);

    rhoR0 = 0.2;
    uR0 = 0.0; vR0 = 0.0; wR0 = 0.0;
    bXR0 = 1.0; bYR0 = 0.0; bZR0 = 0.0;
    pR0 = 0.1;
    eR0 = pR0 / (gamma_mhd - 1.0)
        + 0.5 * rhoR0 * (uR0 * uR0 + vR0 * vR0 + wR0 * wR0)
        + 0.5 * (bXR0 * bXR0 + bYR0 * bYR0 + bZR0 * bZR0);
    

    int threadsPerBlock = 256;
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        rhoL0, uL0, vL0, wL0, bXL0, bYL0, bZL0, pL0, eL0, 
        rhoR0, uR0, vR0, wR0, bXR0, bYR0, bZR0, pR0, eR0
    );

    cudaDeviceSynchronize();
}


int main()
{
    initializeDeviceConstants();

    std::string directoryname = "results";
    std::string filenameWithoutStep = "shock_tube";
    std::ofstream logfile("log.txt");
    int recordStep = 100;

    IdealMHD1D idealMHD1D;

    idealMHD1D.initializeU();

    for (int step = 0; step < totalStep+1; step++) {
        if (step % recordStep == 0) {
            idealMHD1D.save(directoryname, filenameWithoutStep, step);
            logfile << std::to_string(step) << ","
                    << std::setprecision(6) << totalTime
                    << std::endl;
        }
        idealMHD1D.oneStepRK2();
        totalTime += dt;
    }
    
    return 0;
}


