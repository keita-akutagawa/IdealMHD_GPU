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


void IdealMHD1D::initializeU()
{
    double rhoL0, uL0, vL0, wL0, bxL0, byL0, bzL0, pL0, eL0;
    double rhoR0, uR0, vR0, wR0, bxR0, byR0, bzR0, pR0, eR0;

    rhoL0 = 1.0;
    uL0 = 10.0; vL0 = 0.0; wL0 = 0.0;
    bxL0 = 5.0 / sqrt(4.0 * PI); byL0 = 5.0 / sqrt(4.0 * PI); bzL0 = 0.0;
    pL0 = 20.0;
    eL0 = pL0 / (gamma_mhd - 1.0)
        + 0.5 * rhoL0 * (uL0 * uL0 + vL0 * vL0 + wL0 * wL0)
        + 0.5 * (bxL0 * bxL0 + byL0 * byL0 + bzL0 * bzL0);

    rhoR0 = 1.0;
    uR0 = -10.0; vR0 = 0.0; wR0 = 0.0;
    bxR0 = 5.0 / sqrt(4.0 * PI); byR0 = 5.0 / sqrt(4.0 * PI); bzR0 = 0.0;
    pR0 = 1.0;
    eR0 = pR0 / (gamma_mhd - 1.0)
        + 0.5 * rhoR0 * (uR0 * uR0 + vR0 * vR0 + wR0 * wR0)
        + 0.5 * (bxR0 * bxR0 + byR0 * byR0 + bzR0 * bzR0);

    for (int i = 0; i < int(nx / 2.0); i++) {
        U[0][i] = rhoL0;
        U[1][i] = rhoL0 * uL0;
        U[2][i] = rhoL0 * vL0;
        U[3][i] = rhoL0 * wL0;
        U[4][i] = bxL0;
        U[5][i] = byL0;
        U[6][i] = bzL0;
        U[7][i] = eL0;
    }
    for (int i = int(nx / 2.0); i < nx; i++) {
        U[0][i] = rhoR0;
        U[1][i] = rhoR0 * uR0;
        U[2][i] = rhoR0 * vR0;
        U[3][i] = rhoR0 * wR0;
        U[4][i] = bxR0;
        U[5][i] = byR0;
        U[6][i] = bzR0;
        U[7][i] = eR0;
    }
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
                    << std::setprecision(3) << totalTime
                    << std::endl;
        }
        idealMHD1D.oneStepRK2();
        totalTime += dt;
    }
    
    return 0;
}


