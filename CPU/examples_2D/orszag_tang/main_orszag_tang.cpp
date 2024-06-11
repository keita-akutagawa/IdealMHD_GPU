#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../lib_IdealMHD_2D/const.hpp"
#include "../../lib_IdealMHD_2D/idealMHD_2D.hpp"

const double EPS = 1e-20;
const double PI = 3.141592653589793;
const double dtError = 1e100;

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
const int totalStep = 10000;
double totalTime = 0.0;


int main()
{
    std::string directoryname = "results_CT";
    std::string filenameWithoutStep = "orszag_tang";
    std::ofstream logfile("log.txt");
    int recordStep = 10;


    double rho0, u0, v0, w0, bx0, by0, bz0, p0, e0;
    
    std::vector<std::vector<std::vector<double>>> UInit(8, std::vector(nx, std::vector<double>(ny, 0.0)));
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            rho0 = gamma_mhd * gamma_mhd;
            u0 = -sin(j * dy);
            v0 = sin(i * dx);
            w0 = 0.0;
            bx0 = -sin(j * dy);
            by0 = sin(2.0 * i * dx);
            bz0 = 0.0;
            p0 = gamma_mhd;
            e0 = p0 / (gamma_mhd - 1.0)
               + 0.5 * rho0 * (u0 * u0 + v0 * v0 + w0 * w0)
               + 0.5 * (bx0 * bx0 + by0 * by0 + bz0 * bz0);

            UInit[0][i][j] = rho0;
            UInit[1][i][j] = rho0 * u0;
            UInit[2][i][j] = rho0 * v0;
            UInit[3][i][j] = rho0 * w0;
            UInit[4][i][j] = bx0;
            UInit[5][i][j] = by0;
            UInit[6][i][j] = bz0;
            UInit[7][i][j] = e0;
        }
    }


    IdealMHD2D idealMHD2D;

    idealMHD2D.initializeU(UInit);

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


