#include "ct.hpp"


void CT::setOldFlux2D(
    const Flux2D& flux2D
)
{
    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                oldFlux2D.fluxF[comp][i][j] = flux2D.fluxF[comp][i][j];
                oldFlux2D.fluxG[comp][i][j] = flux2D.fluxG[comp][i][j];
            }
        }
    }
}


void CT::divBClean(
    const std::vector<std::vector<double>>& bxOld,
    const std::vector<std::vector<double>>& byOld, 
    std::vector<std::vector<std::vector<double>>>& U
)
{
    double ezF1, ezF2, ezG1, ezG2, ez;
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny-1; j++) {
            ezG1 = oldFlux2D.fluxG[4][i][j];
            ezG2 = oldFlux2D.fluxG[4][i+1][j];
            ezF1 = -1.0 * oldFlux2D.fluxF[5][i][j];
            ezF2 = -1.0 * oldFlux2D.fluxF[5][i][j+1];
            ez = 0.25 * (ezG1 + ezG2 + ezF1 + ezF2);
            EzVector[i][j] = ez;
        }
    }

    for (int i = 0; i < nx-1; i++) {
        ezG1 = oldFlux2D.fluxG[4][i][ny-1];
        ezG2 = oldFlux2D.fluxG[4][i+1][ny-1];
        ezF1 = -1.0 * oldFlux2D.fluxF[5][i][ny-1];
        ezF2 = -1.0 * oldFlux2D.fluxF[5][i][0];
        ez = 0.25 * (ezG1 + ezG2 + ezF1 + ezF2);
        EzVector[i][ny-1] = ez;
    }

    for (int j = 0; j < ny-1; j++) {
        ezG1 = oldFlux2D.fluxG[4][nx-1][j];
        ezG2 = oldFlux2D.fluxG[4][0][j];
        ezF1 = -1.0 * oldFlux2D.fluxF[5][nx-1][j];
        ezF2 = -1.0 * oldFlux2D.fluxF[5][nx-1][j+1];
        ez = 0.25 * (ezG1 + ezG2 + ezF1 + ezF2);
        EzVector[nx-1][j] = ez;
    }

    ezG1 = oldFlux2D.fluxG[4][nx-1][ny-1];
    ezG2 = oldFlux2D.fluxG[4][0][ny-1];
    ezF1 = -1.0 * oldFlux2D.fluxF[5][nx-1][ny-1];
    ezF2 = -1.0 * oldFlux2D.fluxF[5][nx-1][0];
    ez = 0.25 * (ezG1 + ezG2 + ezF1 + ezF2);
    EzVector[nx-1][ny-1] = ez;


    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            U[4][i][j] = bxOld[i][j]
                       - dt / dy * (EzVector[i][j] - EzVector[i][j-1]);
            U[5][i][j] = byOld[i][j]
                       + dt / dx * (EzVector[i][j] - EzVector[i-1][j]);
        }
    }

    for (int i = 1; i < nx; i++) {
        U[4][i][0] = bxOld[i][0]
                    - dt / dy * (EzVector[i][0] - EzVector[i][ny-1]);
        U[5][i][0] = byOld[i][0]
                    + dt / dx * (EzVector[i][0] - EzVector[i-1][0]);
    }

    for (int j = 1; j < ny; j++) {
        U[4][0][j] = bxOld[0][j]
                    - dt / dy * (EzVector[0][j] - EzVector[0][j-1]);
        U[5][0][j] = byOld[0][j]
                    + dt / dx * (EzVector[0][j] - EzVector[nx-1][j]);
    }

    U[4][0][0] = bxOld[0][0]
                - dt / dy * (EzVector[0][0] - EzVector[0][ny-1]);
    U[5][0][0] = byOld[0][0]
                + dt / dx * (EzVector[0][0] - EzVector[nx-1][0]);

}

