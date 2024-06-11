#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../../lib_IdealMHD_1D/const.hpp"
#include "../../../lib_IdealMHD_1D/idealMHD_1D.hpp"

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


int main()
{
    std::string directoryname = "results";
    std::string filenameWithoutStep = "shock_tube";
    std::ofstream logfile("log.txt");
    int recordStep = 100;


    double rhoL0, uL0, vL0, wL0, bxL0, byL0, bzL0, pL0, eL0;
    double rhoR0, uR0, vR0, wR0, bxR0, byR0, bzR0, pR0, eR0;

    rhoL0 = 1.08;
    uL0 = 1.2; vL0 = 0.01; wL0 = 0.5;
    bxL0 = 2.0 / sqrt(4.0 * PI); byL0 = 3.6 / sqrt(4.0 * PI); bzL0 = 2.0 / sqrt(4.0 * PI);
    pL0 = 0.95;
    eL0 = pL0 / (gamma_mhd - 1.0)
        + 0.5 * rhoL0 * (uL0 * uL0 + vL0 * vL0 + wL0 * wL0)
        + 0.5 * (bxL0 * bxL0 + byL0 * byL0 + bzL0 * bzL0);

    rhoR0 = 1.0;
    uR0 = 0.0; vR0 = 0.0; wR0 = 0.0;
    bxR0 = 2.0 / sqrt(4.0 * PI); byR0 = 4.0 / sqrt(4.0 * PI); bzR0 = 2.0 / sqrt(4.0 * PI);
    pR0 = 1.0;
    eR0 = pR0 / (gamma_mhd - 1.0)
        + 0.5 * rhoR0 * (uR0 * uR0 + vR0 * vR0 + wR0 * wR0)
        + 0.5 * (bxR0 * bxR0 + byR0 * byR0 + bzR0 * bzR0);

    std::vector<std::vector<double>> UInit(8, std::vector<double>(nx, 0.0));
    for (int i = 0; i < int(nx / 2.0); i++) {
        UInit[0][i] = rhoL0;
        UInit[1][i] = rhoL0 * uL0;
        UInit[2][i] = rhoL0 * vL0;
        UInit[3][i] = rhoL0 * wL0;
        UInit[4][i] = bxL0;
        UInit[5][i] = byL0;
        UInit[6][i] = bzL0;
        UInit[7][i] = eL0;
    }
    for (int i = int(nx / 2.0); i < nx; i++) {
        UInit[0][i] = rhoR0;
        UInit[1][i] = rhoR0 * uR0;
        UInit[2][i] = rhoR0 * vR0;
        UInit[3][i] = rhoR0 * wR0;
        UInit[4][i] = bxR0;
        UInit[5][i] = byR0;
        UInit[6][i] = bzR0;
        UInit[7][i] = eR0;
    }


    IdealMHD1D idealMHD1D;

    idealMHD1D.initializeU(UInit);

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


