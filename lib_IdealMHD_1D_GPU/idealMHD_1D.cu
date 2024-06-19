#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "const.hpp"
#include "idealMHD_1D.hpp"


struct oneStepFirstFunctor {

    __device__
    ConservationParameter operator()(
        const ConservationParameter& u, 
        const Flux& fluxF, 
        const Flux& fluxFMinus1
    ) const {
        ConservationParameter uNext;

        uNext.rho  = u.rho  - device_dt / device_dx * (fluxF.f0 - fluxFMinus1.f0);
        uNext.rhoU = u.rhoU - device_dt / device_dx * (fluxF.f1 - fluxFMinus1.f1);
        uNext.rhoV = u.rhoV - device_dt / device_dx * (fluxF.f2 - fluxFMinus1.f2);
        uNext.rhoW = u.rhoW - device_dt / device_dx * (fluxF.f3 - fluxFMinus1.f3);
        uNext.bX   = u.bX   - device_dt / device_dx * (fluxF.f4 - fluxFMinus1.f4);
        uNext.bY   = u.bY   - device_dt / device_dx * (fluxF.f5 - fluxFMinus1.f5);
        uNext.bZ   = u.bZ   - device_dt / device_dx * (fluxF.f6 - fluxFMinus1.f6);
        uNext.e    = u.e    - device_dt / device_dx * (fluxF.f7 - fluxFMinus1.f7);

        return uNext;
    }
};

struct oneStepSecondFunctor {

    __device__
    ConservationParameter operator()(
        const ConservationParameter& u, 
        const ConservationParameter& uBar, 
        const Flux& fluxF, 
        const Flux& fluxFMinus1
    ) const {
        ConservationParameter uNext;

        uNext.rho   = 0.5 * (u.rho + uBar.rho
                    - device_dt / device_dx * (fluxF.f0 - fluxFMinus1.f0));
        uNext.rhoU  = 0.5 * (u.rhoU + uBar.rhoU
                    - device_dt / device_dx * (fluxF.f1 - fluxFMinus1.f1));
        uNext.rhoV  = 0.5 * (u.rhoV + uBar.rhoV
                    - device_dt / device_dx * (fluxF.f2 - fluxFMinus1.f2));
        uNext.rhoW  = 0.5 * (u.rhoW + uBar.rhoW
                    - device_dt / device_dx * (fluxF.f3 - fluxFMinus1.f3));
        uNext.bX    = 0.5 * (u.bX + uBar.bX
                    - device_dt / device_dx * (fluxF.f4 - fluxFMinus1.f4));
        uNext.bY    = 0.5 * (u.bY + uBar.bY
                    - device_dt / device_dx * (fluxF.f5 - fluxFMinus1.f5));
        uNext.bZ    = 0.5 * (u.bZ + uBar.bZ
                    - device_dt / device_dx * (fluxF.f6 - fluxFMinus1.f6));
        uNext.e     = 0.5 * (u.e + uBar.e
                    - device_dt / device_dx * (fluxF.f7 - fluxFMinus1.f7));

        return uNext;
    }
};


void IdealMHD1D::oneStepRK2()
{
    thrust::copy(U.begin(), U.end(), UBar.begin());

    calculateDt();

    fluxF = fluxSolver.getFluxF(U);

    auto tupleForFlux = thrust::make_tuple(U.begin(), fluxF.begin(), fluxF.begin() - 1);
    auto tupleForFluxIterator = thrust::make_zip_iterator(tupleForFlux);
    thrust::transform(
        tupleForFluxIterator + 1, 
        tupleForFluxIterator + nx, 
        UBar.begin() + 1, 
        oneStepFirstFunctor()
    );

    //これはどうにかすること。保守性が低い
    boundary.symmetricBoundary2nd(UBar);

    fluxF = fluxSolver.getFluxF(UBar);

    auto tupleForFlux = thrust::make_tuple(U.begin(), UBar.begin(), fluxF.begin(), fluxF.begin() - 1);
    auto tupleForFluxIterator = thrust::make_zip_iterator(tupleForFlux);
    thrust::transform(
        tupleForFluxIterator + 1, 
        tupleForFluxIterator + nx, 
        U.begin() + 1, 
        oneStepFirstFunctor()
    );

    //これはどうにかすること。保守性が低い
    boundary.symmetricBoundary2nd(U);
}


void IdealMHD1D::save(
    std::string directoryname, 
    std::string filenameWithoutStep, 
    int step
)
{
    hU = U;

    std::string filename;
    filename = directoryname + "/"
             + filenameWithoutStep + "_" + std::to_string(step)
             + ".txt";

    std::ofstream ofs(filename);
    ofs << std::fixed << std::setprecision(6);

    for (int i = 0; i < nx - 1; i++) {
        ofs << hU[i].rho << ',' 
            << hU[i].rhoU << ',' 
            << hU[i].rhoV << ','
            << hU[i].rhoW << ','
            << hU[i].bX << ','
            << hU[i].bY << ','
            << hU[i].bZ << ','
            << hU[i].e << '\n';
    }
    ofs << hU[nx - 1].rho << ',' 
        << hU[nx - 1].rhoU << ',' 
        << hU[nx - 1].rhoV << ','
        << hU[nx - 1].rhoW << ','
        << hU[nx - 1].bX << ','
        << hU[nx - 1].bY << ','
        << hU[nx - 1].bZ << ','
        << hU[nx - 1].e;
}


/*
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
*/


// getter
thrust::device_vector<ConservationParameter> IdealMHD1D::getU()
{
    return U;
}

