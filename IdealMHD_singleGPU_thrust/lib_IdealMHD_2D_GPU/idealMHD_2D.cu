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
      tmpU(nx * ny), 
      dtVector(nx * ny),
      bXOld(nx * ny), 
      bYOld(nx * ny), 
      tmpVector(nx * ny), 
      hU(nx * ny)
{
}



__global__ void copyBX_kernel(
    double* tmp, 
    const ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        tmp[j + i * device_ny] = U[j + i * device_ny].bX;
    }
}

__global__ void copyBY_kernel(
    double* tmp, 
    const ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        tmp[j + i * device_ny] = U[j + i * device_ny].bY;
    }
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
                    - device_dt / device_dx * (fluxF.f0 - fluxFMinusNy.f0)
                    - device_dt / device_dy * (fluxG.f0 - fluxGMinus1.f0));
        UNext.rhoU  = 0.5 * (U.rhoU + UBar.rhoU
                    - device_dt / device_dx * (fluxF.f1 - fluxFMinusNy.f1)
                    - device_dt / device_dy * (fluxG.f1 - fluxGMinus1.f1));
        UNext.rhoV  = 0.5 * (U.rhoV + UBar.rhoV
                    - device_dt / device_dx * (fluxF.f2 - fluxFMinusNy.f2)
                    - device_dt / device_dy * (fluxG.f2 - fluxGMinus1.f2));
        UNext.rhoW  = 0.5 * (U.rhoW + UBar.rhoW
                    - device_dt / device_dx * (fluxF.f3 - fluxFMinusNy.f3)
                    - device_dt / device_dy * (fluxG.f3 - fluxGMinus1.f3));
        UNext.bX    = 0.5 * (U.bX + UBar.bX
                    - device_dt / device_dx * (fluxF.f4 - fluxFMinusNy.f4)
                    - device_dt / device_dy * (fluxG.f4 - fluxGMinus1.f4));
        UNext.bY    = 0.5 * (U.bY + UBar.bY
                    - device_dt / device_dx * (fluxF.f5 - fluxFMinusNy.f5)
                    - device_dt / device_dy * (fluxG.f5 - fluxGMinus1.f5));
        UNext.bZ    = 0.5 * (U.bZ + UBar.bZ
                    - device_dt / device_dx * (fluxF.f6 - fluxFMinusNy.f6)
                    - device_dt / device_dy * (fluxG.f6 - fluxGMinus1.f6));
        UNext.e     = 0.5 * (U.e + UBar.e
                    - device_dt / device_dx * (fluxF.f7 - fluxFMinusNy.f7)
                    - device_dt / device_dy * (fluxG.f7 - fluxGMinus1.f7));

        return UNext;
    }
};


void IdealMHD2D::oneStepRK2()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bYOld.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    thrust::copy(U.begin(), U.end(), UBar.begin());

    calculateDt();

    shiftUToCenterForCT(U);
    fluxF = fluxSolver.getFluxF(U);
    fluxG = fluxSolver.getFluxG(U);
    backUToCenterHalfForCT(U);

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

    ct.setOldFlux2D(fluxF, fluxG);
    ct.divBClean(bXOld, bYOld, UBar);

    //これはどうにかすること。保守性が低い
    boundary.periodicBoundaryX2nd(UBar);
    boundary.periodicBoundaryY2nd(UBar);

    shiftUToCenterForCT(UBar);
    fluxF = fluxSolver.getFluxF(UBar);
    fluxG = fluxSolver.getFluxG(UBar);
    backUToCenterHalfForCT(UBar);

    thrust::copy(U.begin(), U.end(), tmpU.begin());
    auto tupleForFluxSecond = thrust::make_tuple(
        tmpU.begin(), UBar.begin(), fluxF.begin(), fluxF.begin() - ny, fluxG.begin(), fluxG.begin() - 1
    );
    auto tupleForFluxSecondIterator = thrust::make_zip_iterator(tupleForFluxSecond);
    thrust::transform(
        tupleForFluxSecondIterator + ny, 
        tupleForFluxSecondIterator + nx * ny, 
        U.begin() + ny, 
        oneStepSecondFunctor()
    );

    ct.divBClean(bXOld, bYOld, U);

    //これはどうにかすること。保守性が低い
    boundary.periodicBoundaryX2nd(U);
    boundary.periodicBoundaryY2nd(U);
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
             + ".bin";

    std::ofstream ofs(filename, std::ios::binary);
    ofs << std::fixed << std::setprecision(6);

    for (int i = 0; i < nx - 1; i++) {
        for (int j = 0; j < ny; j++) {
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * ny].rho), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * ny].rhoU), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * ny].rhoV), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * ny].rhoW), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * ny].bX), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * ny].bY), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * ny].bZ), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * ny].e), sizeof(double));
        }
    }
    for (int j = 0; j < ny - 1; j++) {
        ofs.write(reinterpret_cast<const char*>(&hU[j + (nx - 1) * ny].rho), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&hU[j + (nx - 1) * ny].rhoU), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&hU[j + (nx - 1) * ny].rhoV), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&hU[j + (nx - 1) * ny].rhoW), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&hU[j + (nx - 1) * ny].bX), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&hU[j + (nx - 1) * ny].bY), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&hU[j + (nx - 1) * ny].bZ), sizeof(double));
        ofs.write(reinterpret_cast<const char*>(&hU[j + (nx - 1) * ny].e), sizeof(double));
    }
    ofs.write(reinterpret_cast<const char*>(&hU[ny - 1 + (nx - 1) * ny].rho), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hU[ny - 1 + (nx - 1) * ny].rhoU), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hU[ny - 1 + (nx - 1) * ny].rhoV), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hU[ny - 1 + (nx - 1) * ny].rhoW), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hU[ny - 1 + (nx - 1) * ny].bX), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hU[ny - 1 + (nx - 1) * ny].bY), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hU[ny - 1 + (nx - 1) * ny].bZ), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hU[ny - 1 + (nx - 1) * ny].e), sizeof(double));
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


struct IsNan
{
    __device__ 
    bool operator()(const ConservationParameter U) const {
        return isnan(U.e); // 何かが壊れたらeは壊れるから
    }
};

bool IdealMHD2D::checkCalculationIsCrashed()
{
    bool result = thrust::transform_reduce(
        U.begin(), U.end(), IsNan(), false, thrust::logical_or<bool>()
    );

    return result;
}

/////////////////////

__global__ void shiftBXToCenterForCT_kernel(
    const double* tmp, 
    ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx) && (j < device_ny)) {
        U[j + i * device_ny].bX = 0.5 * (tmp[j + i * device_ny] + tmp[j + (i - 1) * device_ny]);
    }
}

__global__ void shiftBYToCenterForCT_kernel(
    const double* tmp, 
    ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < device_nx) && (0 < j) && (j < device_ny)) {
        U[j + i * device_ny].bY = 0.5 * (tmp[j + i * device_ny] + tmp[j - 1 + i * device_ny]);
    }
}


void IdealMHD2D::shiftUToCenterForCT(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);


    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    
    shiftBXToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data())
    );

    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    
    shiftBYToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data())
    );
}


__global__ void backBXToCenterForCT_kernel(
    const double* tmp, 
    ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < device_nx - 1) && (j < device_ny)) {
        U[j + i * device_ny].bX = 0.5 * (tmp[j + i * device_ny] + tmp[j + (i + 1) * device_ny]);
    }
}

__global__ void backBYToCenterForCT_kernel(
    const double* tmp, 
    ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < device_nx) && (j < device_ny - 1)) {
        U[j + i * device_ny].bY = 0.5 * (tmp[j + i * device_ny] + tmp[j + 1 + i * device_ny]);
    }
}


void IdealMHD2D::backUToCenterHalfForCT(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);


    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    
    backBXToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data())
    );

    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data())
    );
    
    backBYToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data())
    );
}



// getter
thrust::device_vector<ConservationParameter> IdealMHD2D::getU()
{
    return U;
}

