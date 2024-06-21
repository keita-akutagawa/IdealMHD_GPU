#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <thrust/extrema.h>
#include "const.hpp"
#include "idealMHD_2D.hpp"


IdealMHD2D::IdealMHD2D()
    : fluxF(nx * ny),
      fluxG(nx * ny),
      U(nx * ny),
      UBar(nx * ny), 
      dtVector(nx * ny),
      hU(nx * ny)
{
}


struct oneStepFirstFunctor {

    __device__
    ConservationParameter operator()(
        const thrust::tuple<ConservationParameter, Flux, Flux, Flux, Flux>& tupleForOneStep1
    ) const 
    {
        ConservationParameter U = thrust::get<0>(tupleForOneStep1);
        Flux fluxF              = thrust::get<1>(tupleForOneStep1);
        Flux fluxFMinusNy       = thrust::get<2>(tupleForOneStep1);
        Flux fluxG              = thrust::get<3>(tupleForOneStep1);
        Flux fluxGMinus1        = thrust::get<4>(tupleForOneStep1);
        ConservationParameter UNext;

        UNext.rho  = U.rho  - device_dt / device_dx * (fluxF.f0 - fluxFMinusNy.f0)
                            - device_dt / device_dy * (fluxG.f0 - fluxGMinus1.f0);
        UNext.rhoU = U.rhoU - device_dt / device_dx * (fluxF.f1 - fluxFMinusNy.f1)
                            - device_dt / device_dy * (fluxG.f1 - fluxGMinus1.f1);
        UNext.rhoV = U.rhoV - device_dt / device_dx * (fluxF.f2 - fluxFMinusNy.f2)
                            - device_dt / device_dy * (fluxG.f2 - fluxGMinus1.f2);
        UNext.rhoW = U.rhoW - device_dt / device_dx * (fluxF.f3 - fluxFMinusNy.f3)
                            - device_dt / device_dy * (fluxG.f3 - fluxGMinus1.f3);
        UNext.bX   = U.bX   - device_dt / device_dx * (fluxF.f4 - fluxFMinusNy.f4)
                            - device_dt / device_dy * (fluxG.f4 - fluxGMinus1.f4);
        UNext.bY   = U.bY   - device_dt / device_dx * (fluxF.f5 - fluxFMinusNy.f5)
                            - device_dt / device_dy * (fluxG.f5 - fluxGMinus1.f5);
        UNext.bZ   = U.bZ   - device_dt / device_dx * (fluxF.f6 - fluxFMinusNy.f6)
                            - device_dt / device_dy * (fluxG.f6 - fluxGMinus1.f6);
        UNext.e    = U.e    - device_dt / device_dx * (fluxF.f7 - fluxFMinusNy.f7)
                            - device_dt / device_dy * (fluxG.f7 - fluxGMinus1.f7);

        return UNext;
    }
};

struct oneStepSecondFunctor {

    __device__
    ConservationParameter operator()(
        const thrust::tuple<ConservationParameter, ConservationParameter, Flux, Flux, Flux, Flux>& tupleForOneStep2
    ) const 
    {
        ConservationParameter U    = thrust::get<0>(tupleForOneStep2);
        ConservationParameter UBar = thrust::get<1>(tupleForOneStep2);
        Flux fluxF                 = thrust::get<2>(tupleForOneStep2);
        Flux fluxFMinusNy          = thrust::get<3>(tupleForOneStep2);
        Flux fluxG                 = thrust::get<4>(tupleForOneStep2);
        Flux fluxGMinus1           = thrust::get<5>(tupleForOneStep2);
        ConservationParameter UNext;

        UNext.rho   = 0.5 * (U.rho + UBar.rho
                    - device_dt / device_dx * (fluxF.f0 - fluxFMinusNy.f0))
                    - device_dt / device_dy * (fluxG.f0 - fluxGMinus1.f0);
        UNext.rhoU  = 0.5 * (U.rhoU + UBar.rhoU
                    - device_dt / device_dx * (fluxF.f1 - fluxGMinus1.f1))
                    - device_dt / device_dy * (fluxG.f1 - fluxGMinus1.f1);
        UNext.rhoV  = 0.5 * (U.rhoV + UBar.rhoV
                    - device_dt / device_dx * (fluxF.f2 - fluxGMinus1.f2))
                    - device_dt / device_dy * (fluxG.f2 - fluxGMinus1.f2);
        UNext.rhoW  = 0.5 * (U.rhoW + UBar.rhoW
                    - device_dt / device_dx * (fluxF.f3 - fluxGMinus1.f3))
                    - device_dt / device_dy * (fluxG.f3 - fluxGMinus1.f3);
        UNext.bX    = 0.5 * (U.bX + UBar.bX
                    - device_dt / device_dx * (fluxF.f4 - fluxGMinus1.f4))
                    - device_dt / device_dy * (fluxG.f4 - fluxGMinus1.f4);
        UNext.bY    = 0.5 * (U.bY + UBar.bY
                    - device_dt / device_dx * (fluxF.f5 - fluxGMinus1.f5))
                    - device_dt / device_dy * (fluxG.f5 - fluxGMinus1.f5);
        UNext.bZ    = 0.5 * (U.bZ + UBar.bZ
                    - device_dt / device_dx * (fluxF.f6 - fluxGMinus1.f6))
                    - device_dt / device_dy * (fluxG.f6 - fluxGMinus1.f6);
        UNext.e     = 0.5 * (U.e + UBar.e
                    - device_dt / device_dx * (fluxF.f7 - fluxGMinus1.f7))
                    - device_dt / device_dy * (fluxG.f7 - fluxGMinus1.f7);

        return UNext;
    }
};


void IdealMHD2D::oneStepRK2()
{
    thrust::copy(U.begin(), U.end(), UBar.begin());

    calculateDt();

    fluxF = fluxSolverF.getFluxF(U);
    fluxG = fluxSolverG.getFluxG(U);

    auto tupleForFluxFirst = thrust::make_tuple(
        U.begin(), fluxF.begin(), fluxF.begin() - ny, fluxG.begin(), fluxG.begin() - 1
    );
    auto tupleForFluxFirstIterator = thrust::make_zip_iterator(tupleForFluxFirst);
    thrust::transform(
        tupleForFluxFirstIterator + ny, 
        tupleForFluxFirstIterator + nx * ny, 
        UBar.begin() + ny, 
        oneStepFirstFunctor()
    );

    //これはどうにかすること。保守性が低い
    boundary.symmetricBoundary2nd(UBar);

    fluxF = fluxSolverF.getFluxF(UBar);
    fluxG = fluxSolverG.getFluxG(UBar);

    auto tupleForFluxSecond = thrust::make_tuple(
        U.begin(), UBar.begin(), fluxF.begin(), fluxF.begin() - ny, fluxG.begin(), fluxG.begin() - 1
    );
    auto tupleForFluxSecondIterator = thrust::make_zip_iterator(tupleForFluxSecond);
    thrust::transform(
        tupleForFluxSecondIterator + ny, 
        tupleForFluxSecondIterator + nx * ny, 
        U.begin() + ny, 
        oneStepSecondFunctor()
    );

    //これはどうにかすること。保守性が低い
    boundary.symmetricBoundary2nd(U);
}


void IdealMHD2D::save(
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
        for (int j = 0; j < ny; j++) {
            ofs << hU[j + i * ny].rho << ',' 
                << hU[j + i * ny].rhoU << ',' 
                << hU[j + i * ny].rhoV << ','
                << hU[j + i * ny].rhoW << ','
                << hU[j + i * ny].bX << ','
                << hU[j + i * ny].bY << ','
                << hU[j + i * ny].bZ << ','
                << hU[j + i * ny].e << '\n';
        }
    }
    for (int j = 0; j < ny; j++) {
        ofs << hU[j + (nx - 1) * ny].rho << ',' 
            << hU[j + (nx - 1) * ny].rhoU << ',' 
            << hU[j + (nx - 1) * ny].rhoV << ','
            << hU[j + (nx - 1) * ny].rhoW << ','
            << hU[j + (nx - 1) * ny].bX << ','
            << hU[j + (nx - 1) * ny].bY << ','
            << hU[j + (nx - 1) * ny].bZ << ','
            << hU[j + (nx - 1) * ny].e;
    }
    
}


struct calculateDtFunctor {

    __device__
    double operator()(const ConservationParameter& conservationParameter) const {
        double rho, u, v, w, bX, bY, bZ, e, p, cs, ca;
        double maxSpeedX, maxSpeedY;

        rho = conservationParameter.rho;
        u = conservationParameter.rhoU / rho;
        v = conservationParameter.rhoV / rho;
        w = conservationParameter.rhoW / rho;
        bX = conservationParameter.bX;
        bY = conservationParameter.bY;
        bZ = conservationParameter.bZ;
        e = conservationParameter.e;
        p = (device_gamma_mhd - 1.0)
          * (e - 0.5 * rho * (u * u + v * v + w * w)
          - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        cs = sqrt(device_gamma_mhd * p / rho);
        ca = sqrt((bX * bX + bY * bY + bZ * bZ) / rho);

        maxSpeedX = std::abs(u) + sqrt(cs * cs + ca * ca);
        maxSpeedY = std::abs(v) + sqrt(cs * cs + ca * ca);

        return 1.0 / (maxSpeedX / device_dx + maxSpeedY / device_dy + device_EPS);
    }
};


void IdealMHD2D::calculateDt()
{
    thrust::transform(
        U.begin(), 
        U.end(), 
        dtVector.begin(), 
        calculateDtFunctor()
    );

    thrust::device_vector<double>::iterator dtMin = thrust::min_element(dtVector.begin(), dtVector.end());
    
    dt = (*dtMin) * CFL;
    cudaMemcpyToSymbol(device_dt, &dt, sizeof(double));
}


// getter
thrust::device_vector<ConservationParameter> IdealMHD2D::getU()
{
    return U;
}

