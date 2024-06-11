#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "idealMHD_2D.hpp"


void IdealMHD2D::initializeU(
    const std::vector<std::vector<std::vector<double>>>& UInit
)
{
    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                U[comp][i][j] = UInit[comp][i][j];
            }
        }
    }
}


void IdealMHD2D::oneStepRK2()
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            bxOld[i][j] = U[4][i][j];
            byOld[i][j] = U[5][i][j];
        }
    }

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                UBar[comp][i][j] = U[comp][i][j];
            }
        }
    }

    calculateDt();

    shiftUToCenterForCT(U);
    flux2D = fluxSolver.getFluxF(U);
    flux2D = fluxSolver.getFluxG(U);
    backUToCenterHalfForCT(U);

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 1; i < nx; i++) {
            for (int j = 1; j < ny; j++) {
                UBar[comp][i][j] = U[comp][i][j]
                                 - dt / dx * (flux2D.fluxF[comp][i][j] - flux2D.fluxF[comp][i-1][j])
                                 - dt / dy * (flux2D.fluxG[comp][i][j] - flux2D.fluxG[comp][i][j-1]);
            }
        }
        //周期境界条件
        for (int i = 1; i < nx; i++) {
            UBar[comp][i][0] = U[comp][i][0]
                             - dt / dx * (flux2D.fluxF[comp][i][0] - flux2D.fluxF[comp][i-1][0])
                             - dt / dy * (flux2D.fluxG[comp][i][0] - flux2D.fluxG[comp][i][ny-1]);
        }
        for (int j = 1; j < ny; j++) {
            UBar[comp][0][j] = U[comp][0][j]
                             - dt / dx * (flux2D.fluxF[comp][0][j] - flux2D.fluxF[comp][nx-1][j])
                             - dt / dy * (flux2D.fluxG[comp][0][j] - flux2D.fluxG[comp][0][j-1]);
        }
        UBar[comp][0][0] = U[comp][0][0]
                         - dt / dx * (flux2D.fluxF[comp][0][0] - flux2D.fluxF[comp][nx-1][0])
                         - dt / dy * (flux2D.fluxG[comp][0][0] - flux2D.fluxG[comp][0][ny-1]);
    }

    ct.setOldFlux2D(flux2D);
    ct.divBClean(bxOld, byOld, UBar);

    //これはどうにかすること。保守性が低い
    boundary.periodicBoundary(UBar);

    shiftUToCenterForCT(UBar);
    flux2D = fluxSolver.getFluxF(UBar);
    flux2D = fluxSolver.getFluxG(UBar);
    backUToCenterHalfForCT(UBar);

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 1; i < nx; i++) {
            for (int j = 1; j < ny; j++) {
                U[comp][i][j] = 0.5 * (U[comp][i][j] + UBar[comp][i][j]
                              - dt / dx * (flux2D.fluxF[comp][i][j] - flux2D.fluxF[comp][i-1][j])
                              - dt / dy * (flux2D.fluxG[comp][i][j] - flux2D.fluxG[comp][i][j-1]));
            }
            
        }
        //周期境界条件
        for (int i = 1; i < nx; i++) {
            U[comp][i][0] = 0.5 * (U[comp][i][0] + UBar[comp][i][0]
                          - dt / dx * (flux2D.fluxF[comp][i][0] - flux2D.fluxF[comp][i-1][0])
                          - dt / dy * (flux2D.fluxG[comp][i][0] - flux2D.fluxG[comp][i][ny-1]));
        }
        for (int j = 1; j < ny; j++) {
            U[comp][0][j] = 0.5 * (U[comp][0][j] + UBar[comp][0][j]
                            - dt / dx * (flux2D.fluxF[comp][0][j] - flux2D.fluxF[comp][nx-1][j])
                            - dt / dy * (flux2D.fluxG[comp][0][j] - flux2D.fluxG[comp][0][j-1]));
        }
        U[comp][0][0] = 0.5 * (U[comp][0][0] + UBar[comp][0][0]
                            - dt / dx * (flux2D.fluxF[comp][0][0] - flux2D.fluxF[comp][nx-1][0])
                            - dt / dy * (flux2D.fluxG[comp][0][0] - flux2D.fluxG[comp][0][ny-1]));
    }

    ct.divBClean(bxOld, byOld, U);

    //これはどうにかすること。保守性が低い
    boundary.periodicBoundary(U);
}


void IdealMHD2D::save(
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
            for (int j = 0; j < ny; j++) {
                ofs << U[comp][i][j] << ",";
            }
        }
        for (int j = 0; j < ny-1; j++) {
            ofs << U[comp][nx-1][j] << ",";
        }
        ofs << U[comp][nx-1][ny-1];
        ofs << std::endl;
    }
}


void IdealMHD2D::calculateDt()
{
    double rho, u, v, w, bx, by, bz, e, p, cs, ca;
    double maxSpeedX, maxSpeedY;
    
    dt = dtError; //十分大きくしておく
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            rho = U[0][i][j];
            u = U[1][i][j] / rho;
            v = U[2][i][j] / rho;
            w = U[3][i][j] / rho;
            bx = U[4][i][j];
            by = U[5][i][j];
            bz = U[6][i][j];
            e = U[7][i][j];
            p = (gamma_mhd - 1.0)
            * (e - 0.5 * rho * (u * u + v * v + w * w)
            - 0.5 * (bx * bx + by * by + bz * bz));
            
            cs = sqrt(gamma_mhd * p / rho);
            ca = sqrt((bx * bx + by * by + bz * bz) / rho);

            maxSpeedX = std::abs(u) + sqrt(cs * cs + ca * ca);
            maxSpeedY = std::abs(v) + sqrt(cs * cs + ca * ca);

            dt = std::min(dt, 1.0 / (maxSpeedX / dx + maxSpeedY / dy + EPS));
        }
    }
    
    dt *= CFL;
}


bool IdealMHD2D::checkCalculationIsCrashed()
{
    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                if (std::isnan(U[comp][i][j])) {
                    return true;
                }
            }
        }
    }

    return false;
}


void IdealMHD2D::shiftUToCenterForCT(
    std::vector<std::vector<std::vector<double>>>& U
)
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            tmpVector[i][j] = U[4][i][j];
        }
    }

    for (int i = 1; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            U[4][i][j] = 0.5 * (tmpVector[i][j] + tmpVector[i-1][j]);
        }
    }
    for (int j = 0; j < ny; j++) {
        U[4][0][j] = 0.5 * (tmpVector[0][j] + tmpVector[nx-1][j]);
    }


    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            tmpVector[i][j] = U[5][i][j];
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            U[5][i][j] = 0.5 * (tmpVector[i][j] + tmpVector[i][j-1]);
        }
    }
    for (int i = 0; i < nx; i++) {
        U[5][i][0] = 0.5 * (tmpVector[i][0] + tmpVector[i][ny-1]);
    }

}

void IdealMHD2D::backUToCenterHalfForCT(
    std::vector<std::vector<std::vector<double>>>& U
)
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            tmpVector[i][j] = U[4][i][j];
        }
    }

    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny; j++) {
            U[4][i][j] = 0.5 * (tmpVector[i][j] + tmpVector[i+1][j]);
        }
    }
    for (int j = 0; j < ny; j++) {
        U[4][nx-1][j] = 0.5 * (tmpVector[nx-1][j] + tmpVector[0][j]);
    }


    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            tmpVector[i][j] = U[5][i][j];
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny-1; j++) {
            U[5][i][j] = 0.5 * (tmpVector[i][j] + tmpVector[i][j+1]);
        }
    }
    for (int i = 0; i < nx; i++) {
        U[5][i][ny-1] = 0.5 * (tmpVector[i][ny-1] + tmpVector[i][0]);
    }

}


// getter
std::vector<std::vector<std::vector<double>>> IdealMHD2D::getU()
{
    return U;
}

