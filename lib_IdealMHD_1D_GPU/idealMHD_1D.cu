#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <thrust/extrema.h>
#include "const.hpp"
#include "idealMHD_1D.hpp"


IdealMHD1D::IdealMHD1D()
    : fluxF(nx),
      U(nx),
      UBar(nx), 
      hU(nx)
{
}


struct oneStepFirstFunctor {

    __device__
    ConservationParameter operator()(const thrust::tuple<ConservationParameter, Flux, Flux>& tupleForOneStep1) const {
        ConservationParameter u = thrust::get<0>(tupleForOneStep1);
        Flux fluxF              = thrust::get<1>(tupleForOneStep1);
        Flux fluxFMinus1        = thrust::get<2>(tupleForOneStep1);
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
    ConservationParameter operator()(const thrust::tuple<ConservationParameter, ConservationParameter, Flux, Flux>& tupleForOneStep2) const {
        ConservationParameter u    = thrust::get<0>(tupleForOneStep2);
        ConservationParameter uBar = thrust::get<1>(tupleForOneStep2);
        Flux fluxF                 = thrust::get<2>(tupleForOneStep2);
        Flux fluxFMinus1           = thrust::get<3>(tupleForOneStep2);
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

    auto tupleForFluxFirst = thrust::make_tuple(U.begin(), fluxF.begin(), fluxF.begin() - 1);
    auto tupleForFluxFirstIterator = thrust::make_zip_iterator(tupleForFluxFirst);
    thrust::transform(
        tupleForFluxFirstIterator + 1, 
        tupleForFluxFirstIterator + nx, 
        UBar.begin() + 1, 
        oneStepFirstFunctor()
    );

    //これはどうにかすること。保守性が低い
    boundary.symmetricBoundary2nd(UBar);

    fluxF = fluxSolver.getFluxF(UBar);

    auto tupleForFluxSecond = thrust::make_tuple(U.begin(), UBar.begin(), fluxF.begin(), fluxF.begin() - 1);
    auto tupleForFluxSecondIterator = thrust::make_zip_iterator(tupleForFluxSecond);
    thrust::transform(
        tupleForFluxSecondIterator + 1, 
        tupleForFluxSecondIterator + nx, 
        U.begin() + 1, 
        oneStepSecondFunctor()
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


struct calculateDtFunctor {

    __device__
    double operator()(const ConservationParameter U) const {
        double rho, u, v, w, bX, bY, bZ, e, p, cs, ca;
        double maxSpeed, dt;

        rho = U.rho;
        u = U.rhoU / rho;
        v = U.rhoV / rho;
        w = U.rhoW / rho;
        bX = U.bX;
        bY = U.bY;
        bZ = U.bZ;
        e = U.e;
        p = (device_gamma_mhd - 1.0)
          * (e - 0.5 * rho * (u * u + v * v + w * w)
          - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        cs = sqrt(gamma_mhd * p / rho);
        ca = sqrt((bX * bX + bY * bY + bZ * bZ) / rho);

        maxSpeed = std::abs(u) + sqrt(cs * cs + ca * ca);
        dt = 1.0 / (maxSpeed / dx + EPS);

        return dt;
    }
};


void IdealMHD1D::calculateDt()
{
    thrust::transform(
        U.begin(), 
        U.end(), 
        dtVector.begin(), 
        calculateDtFunctor()
    );

    thrust::device_vector<double>::iterator dtMin = thrust::min_element(dtVector.begin(), dtVector.end());
    
    dt =  *dtMin * CFL;
}


// getter
thrust::device_vector<ConservationParameter> IdealMHD1D::getU()
{
    return U;
}

