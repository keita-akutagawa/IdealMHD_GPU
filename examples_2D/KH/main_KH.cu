#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../lib_IdealMHD_2D_GPU_periodicX_symmetricY/const.hpp"
#include "../../lib_IdealMHD_2D_GPU_periodicX_symmetricY/idealMHD_2D_periodicX_symmetricY.hpp"


std::string directoryname = "results";
std::string filenameWithoutStep = "KH";
std::ofstream logfile("log_KH.txt");

const double EPS = 1e-20;
const double PI = 3.141592653589793;

const double gamma_mhd = 5.0 / 3.0;

const double host_shear_thickness = 1.0;
const double shear_thickness = 1.0;
const double rr = 0.2;
const double br = 1.0;
const double beta = 2.0;
const double theta = device_PI / 2.0;
const double rho0 = 1.0;
const double b0 = 1.0;
const double p0 = beta * b0 * b0 / 2.0;
const double v0 = sqrt(b0 * b0 / rho0 + device_gamma_mhd * p0 / rho0);

const double xmin = 0.0;
const double xmax = 2.0 * PI * host_shear_thickness / 0.4;
const double dx = host_shear_thickness / 8.0;
const int nx = int((xmax - xmin) / dx);
const double ymin = 0.0;
const double ymax = 2.0 * 10.0 * host_shear_thickness;
const double dy = host_shear_thickness / 8.0;
const int ny = int((ymax - ymin) / dy);

const double xCenter = (xmax - xmin) / 2.0;
const double yCenter = (ymax - ymin) / 2.0;

const double CFL = 0.7;
double dt = 0.0;
const int totalStep = 10000;
const int recordStep = 100;
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

__constant__ double device_xCenter;
__constant__ double device_yCenter;

__constant__ double device_CFL;
__constant__ double device_gamma_mhd;

__device__ double device_dt;

__constant__ double device_shear_thickness;
__constant__ double device_rr;
__constant__ double device_br;
__constant__ double device_beta;
__constant__ double device_theta;
__constant__ double device_rho0;
__constant__ double device_b0;
__constant__ double device_p0;
__constant__ double device_v0;


__global__ void initializeU_kernel(ConservationParameter* U) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        double xPosition, yPosition;
        xPosition = i * device_dx - device_xCenter;
        yPosition = j * device_dy - device_yCenter;

        double rho, u, v, w, bx, by, bz, p, e;
        rho = device_rho0 / 2.0 * ((1.0 - device_rr) * tanh(yPosition / device_shear_thickness) + 1.0 + device_rr);
        u = -device_v0 / 2.0 * tanh(yPosition / device_shear_thickness);
        v = 0.02 * device_v0 * cos(2.0 * device_PI * xPosition / device_xmax) / pow(cosh(yPosition / device_shear_thickness), 2);
        w = 0.0;
        bx = device_b0 / 2.0 * ((1.0 - device_br) * tanh(yPosition / device_shear_thickness) + 1.0 + device_br) * cos(device_theta);
        by = 0.0;
        bz = device_b0 / 2.0 * ((1.0 - device_br) * tanh(yPosition / device_shear_thickness) + 1.0 + device_br) * sin(device_theta);
        p = device_beta * (bx * bx + by * by + bz * bz) / 2.0;
        e = p / (device_gamma_mhd - 1.0)
            + 0.5 * rho * (u * u + v * v + w * w)
            + 0.5 * (bx * bx + by * by + bz * bz);
        
        U[j + i * device_ny].rho  = rho;
        U[j + i * device_ny].rhoU = rho * u;
        U[j + i * device_ny].rhoV = rho * v;
        U[j + i * device_ny].rhoW = rho * w;
        U[j + i * device_ny].bX   = bx;
        U[j + i * device_ny].bY   = by;
        U[j + i * device_ny].bZ   = bz;
        U[j + i * device_ny].e    = e;
    }
}

void IdealMHD2DPeriodicXSymmetricY::initializeU()
{
    cudaMemcpyToSymbol(device_xCenter, &xCenter, sizeof(double));
    cudaMemcpyToSymbol(device_yCenter, &yCenter, sizeof(double));
    cudaMemcpyToSymbol(device_shear_thickness, &shear_thickness, sizeof(double));
    cudaMemcpyToSymbol(device_rr, &rr, sizeof(double));
    cudaMemcpyToSymbol(device_br, &br, sizeof(double));
    cudaMemcpyToSymbol(device_beta, &beta, sizeof(double));
    cudaMemcpyToSymbol(device_theta, &theta, sizeof(double));
    cudaMemcpyToSymbol(device_rho0, &rho0, sizeof(double));
    cudaMemcpyToSymbol(device_b0, &b0, sizeof(double));
    cudaMemcpyToSymbol(device_p0, &p0, sizeof(double));
    cudaMemcpyToSymbol(device_v0, &v0, sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


int main()
{
    initializeDeviceConstants();

    IdealMHD2DPeriodicXSymmetricY idealMHD2DPeriodicXSymmetricY;

    idealMHD2DPeriodicXSymmetricY.initializeU();

    for (int step = 0; step < totalStep+1; step++) {
        if (step % recordStep == 0) {
            idealMHD2DPeriodicXSymmetricY.save(directoryname, filenameWithoutStep, step);
            logfile << std::to_string(step) << ","
                    << std::setprecision(4) << totalTime
                    << std::endl;
            std::cout << std::to_string(step) << " step done : total time is "
                      << std::setprecision(4) << totalTime
                      << std::endl;
        }
        
        idealMHD2DPeriodicXSymmetricY.oneStepRK2();

        if (idealMHD2DPeriodicXSymmetricY.checkCalculationIsCrashed()) {
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            return 0;
        }

        totalTime += dt;
    }
    
    return 0;
}


