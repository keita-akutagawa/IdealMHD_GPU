#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../lib_IdealMHD_2D_GPU/const.hpp"
#include "../../lib_IdealMHD_2D_GPU/idealMHD_2D.hpp"

const double EPS = 1e-20;
const double PI = 3.141592653589793;

const int nx = 256;
const double xmin = 0.0;
const double xmax = 2.0 * PI;
const double dx = (xmax - xmin) / nx;
const int ny = 256;
const double ymin = 0.0;
const double ymax = 2.0 * PI;
const double dy = (ymax - ymin) / ny;

const double CFL = 0.7;
const double gamma_mhd = 5.0 / 3.0;
double dt = 0.0;
const int totalStep = 10;
double totalTime = 0.0;

__constant__ double device_EPS;
__constant__ double device_PI;

__constant__ double device_dx;
__constant__ double device_xmin;
__constant__ double device_xmax;
__constant__ int device_nx;

__constant__ double device_dy;
__constant__ double device_ymin;
__constant__ double device_ymax;
__constant__ int device_ny;

__constant__ double device_CFL;
__constant__ double device_gamma_mhd;

__device__ double device_dt;

__constant__ int device_totalStep;
__device__ double device_totalTime;


__global__ void initializeU_kernel(ConservationParameter* U) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        double rho0 = device_gamma_mhd * device_gamma_mhd;
        double u0 = -sin(j * device_dy);
        double v0 = sin(i * device_dx);
        double w0 = 0.0;
        double bx0 = -sin(j * device_dy);
        double by0 = sin(2.0 * i * device_dx);
        double bz0 = 0.0;
        double p0 = device_gamma_mhd;
        double e0 = p0 / (device_gamma_mhd - 1.0)
                 + 0.5 * rho0 * (u0 * u0 + v0 * v0 + w0 * w0)
                 + 0.5 * (bx0 * bx0 + by0 * by0 + bz0 * bz0);
        
        U[j + i * ny].rho  = rho0;
        U[j + i * ny].rhoU = rho0 * u0;
        U[j + i * ny].rhoV = rho0 * v0;
        U[j + i * ny].rhoW = rho0 * w0;
        U[j + i * ny].bX   = bx0;
        U[j + i * ny].bY   = by0;
        U[j + i * ny].bZ   = bz0;
        U[j + i * ny].e    = e0;
    }
}

void IdealMHD2D::initializeU()
{

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


int main()
{
    initializeDeviceConstants();

    std::string directoryname = "results";
    std::string filenameWithoutStep = "orszag_tang";
    std::ofstream logfile("log.txt");
    int recordStep = 1;


    IdealMHD2D idealMHD2D;

    idealMHD2D.initializeU();

    for (int step = 0; step < totalStep+1; step++) {
        if (step % recordStep == 0) {
            idealMHD2D.save(directoryname, filenameWithoutStep, step);
            logfile << std::to_string(step) << ","
                    << std::setprecision(4) << totalTime
                    << std::endl;
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << totalTime
                      << std::endl;
        }
        
        idealMHD2D.oneStepRK2();

        if (idealMHD2D.checkCalculationIsCrashed()) {
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            return 0;
        }

        totalTime += dt;
    }
    
    return 0;
}


