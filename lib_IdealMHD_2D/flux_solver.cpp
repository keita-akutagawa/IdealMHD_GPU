#include "flux_solver.hpp"


Flux2D FluxSolver::getFluxF(
    const std::vector<std::vector<std::vector<double>>>& U
)
{
    for (int j = 0; j < ny; j++) {
        for (int comp = 0; comp < 8; comp++) {
            for (int i = 0; i < nx; i++) {
                tmpUForF[comp][i] = U[comp][i][j];
            }
        }

        hLLDForF.calculateFlux(tmpUForF);
        flux1DForF = hLLDForF.getFlux();

        for (int comp = 0; comp < 8; comp++) {
            for (int i = 0; i < nx; i++) {
                flux2D.fluxF[comp][i][j] = flux1DForF.flux[comp][i];
            }
        }
    }

    return flux2D;
}


//fluxGはfluxFの計算で使う変数を入れ替えることで計算する
Flux2D FluxSolver::getFluxG(
    const std::vector<std::vector<std::vector<double>>>& U
)
{
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            tmpUForG[0][j] = U[0][i][j];
            tmpUForG[1][j] = U[2][i][j];
            tmpUForG[2][j] = U[3][i][j];
            tmpUForG[3][j] = U[1][i][j];
            tmpUForG[4][j] = U[5][i][j];
            tmpUForG[5][j] = U[6][i][j];
            tmpUForG[6][j] = U[4][i][j];
            tmpUForG[7][j] = U[7][i][j];
        }

        hLLDForG.calculateFlux(tmpUForG);
        flux1DForG = hLLDForG.getFlux();

        for (int comp = 0; comp < 8; comp++) {
            for (int j = 0; j < ny; j++) {
                flux2D.fluxG[comp][i][j] = flux1DForG.flux[comp][j];
            }
        }
    }

    setFluxGToProperPosition();

    return flux2D;
}


void FluxSolver::setFluxGToProperPosition()
{
    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                tmpFlux[comp][i][j] = flux2D.fluxG[comp][i][j];
            }
        }
    }
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            flux2D.fluxG[1][i][j] = tmpFlux[3][i][j];
            flux2D.fluxG[2][i][j] = tmpFlux[1][i][j];
            flux2D.fluxG[3][i][j] = tmpFlux[2][i][j];
            flux2D.fluxG[4][i][j] = tmpFlux[6][i][j];
            flux2D.fluxG[5][i][j] = tmpFlux[4][i][j];
            flux2D.fluxG[6][i][j] = tmpFlux[5][i][j];
        }
    }
}
