#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "const.hpp"
#include "idealMHD_1D.hpp"


IdealMHD1D::IdealMHD1D()
{
    U = std::vector(8, std::vector<double>(nx, 0.0));
    UBar = std::vector(8, std::vector<double>(nx, 0.0));
}


void IdealMHD1D::initializeU(
    const std::vector<std::vector<double>> UInit
)
{
    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            U[comp][i] = UInit[comp][i];
        }
    }
}


void IdealMHD1D::oneStepRK2()
{
    for (int comp = 0; comp < 8; comp++) {
        std::copy(U[comp].begin(), U[comp].end(), UBar[comp].begin());
    }

    calculateDt();

    fluxF = fluxSolver.getFluxF(U);

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 1; i < nx; i++) {
            UBar[comp][i] = U[comp][i]
                          - dt / dx * (fluxF.flux[comp][i] - fluxF.flux[comp][i-1]);
        }
        //周期境界条件
        UBar[comp][0] = U[comp][0] 
                      - dt / dx * (fluxF.flux[comp][0] - fluxF.flux[comp][nx-1]);
    }

    //これはどうにかすること。保守性が低い
    boundary.symmetricBoundary2nd(UBar);

    fluxF = fluxSolver.getFluxF(UBar);

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 1; i < nx; i++) {
            U[comp][i] = 0.5 * (U[comp][i] + UBar[comp][i]
                       - dt / dx * (fluxF.flux[comp][i] - fluxF.flux[comp][i-1]));
        }
        //周期境界条件
        U[comp][0] = 0.5 * (U[comp][0] + UBar[comp][0]
                   - dt / dx * (fluxF.flux[comp][0] - fluxF.flux[comp][nx-1]));
    }

    //これはどうにかすること。保守性が低い
    boundary.symmetricBoundary2nd(U);
}


void IdealMHD1D::save(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    std::string filename;
    filename = directoryname + "/"
             + filenameWithoutStep + "_" + std::to_string(step)
             + ".txt";

    std::ofstream ofs(filename);
    ofs << std::fixed << std::setprecision(6);

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx-1; i++) {
            ofs << U[comp][i] << ",";
        }
        ofs << U[comp][nx-1];
        ofs << std::endl;
    }
}


void IdealMHD1D::calculateDt()
{
    double rho, u, v, w, bx, by, bz, e, p, cs, ca;
    double maxSpeed;
    
    dt = 1e100; //十分大きくしておく
    for (int i = 0; i < nx; i++) {
        rho = U[0][i];
        u = U[1][i] / rho;
        v = U[2][i] / rho;
        w = U[3][i] / rho;
        bx = U[4][i];
        by = U[5][i];
        bz = U[6][i];
        e = U[7][i];
        p = (gamma_mhd - 1.0)
          * (e - 0.5 * rho * (u * u + v * v + w * w)
          - 0.5 * (bx * bx + by * by + bz * bz));
        
        cs = sqrt(gamma_mhd * p / rho);
        ca = sqrt((bx * bx + by * by + bz * bz) / rho);

        maxSpeed = std::abs(u) + sqrt(cs * cs + ca * ca);

        dt = std::min(dt, 1.0 / (maxSpeed / dx + EPS));
    }
    
    dt *= CFL;
}


// getter
std::vector<std::vector<double>> IdealMHD1D::getU()
{
    return U;
}

