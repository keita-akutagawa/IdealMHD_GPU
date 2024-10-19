#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <thrust/extrema.h>
#include "const.hpp"
#include "idealMHD2D.hpp"


IdealMHD2D::IdealMHD2D(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      fluxSolver(mPIInfo), 

      fluxF    (mPIInfo.localSizeX * mPIInfo.localSizeY),
      fluxG    (mPIInfo.localSizeX * mPIInfo.localSizeY),
      U        (mPIInfo.localSizeX * mPIInfo.localSizeY),
      UBar     (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      dtVector (mPIInfo.localSizeX * mPIInfo.localSizeY),
      bXOld    (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      bYOld    (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      tmpVector(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      hU       (mPIInfo.localSizeX * mPIInfo.localSizeY), 

      boundary(mPIInfo), 
      ct(mPIInfo)
{

    cudaMalloc(&device_mPIInfo, sizeof(MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(MPIInfo), cudaMemcpyHostToDevice);
    
}



__global__ void copyBX_kernel(
    double* tmp, 
    const ConservationParameter* U, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < localSizeY) {
        tmp[j + i * localSizeY] = U[j + i * localSizeY].bX;
    }
}

__global__ void copyBY_kernel(
    double* tmp, 
    const ConservationParameter* U, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < localSizeY) {
        tmp[j + i * localSizeY] = U[j + i * localSizeY].bY;
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
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_U(U, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
    boundary.periodicBoundaryX2nd(U);
    boundary.periodicBoundaryY2nd(U);
    MPI_Barrier(MPI_COMM_WORLD);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bYOld.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
    thrust::copy(U.begin(), U.end(), UBar.begin());

    calculateDt();

    shiftUToCenterForCT(U);
    fluxF = fluxSolver.getFluxF(U);
    fluxG = fluxSolver.getFluxG(U);
    backUToCenterHalfForCT(U);

    auto tupleForFluxFirst = thrust::make_tuple(
        U.begin(), fluxF.begin(), fluxF.begin() - localSizeY, fluxG.begin(), fluxG.begin() - 1
    );
    auto tupleForFluxFirstIterator = thrust::make_zip_iterator(tupleForFluxFirst);
    thrust::transform(
        tupleForFluxFirstIterator + localSizeY, 
        tupleForFluxFirstIterator + localSizeX * localSizeY, 
        UBar.begin() + localSizeY, 
        oneStepFirstFunctor()
    );

    ct.setOldFlux2D(fluxF, fluxG);
    ct.divBClean(bXOld, bYOld, UBar);

    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_U(UBar, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
    boundary.periodicBoundaryX2nd(UBar);
    boundary.periodicBoundaryY2nd(UBar);
    MPI_Barrier(MPI_COMM_WORLD);

    shiftUToCenterForCT(UBar);
    fluxF = fluxSolver.getFluxF(UBar);
    fluxG = fluxSolver.getFluxG(UBar);
    backUToCenterHalfForCT(UBar);

    auto tupleForFluxSecond = thrust::make_tuple(
        U.begin(), UBar.begin(), fluxF.begin(), fluxF.begin() - localSizeY, fluxG.begin(), fluxG.begin() - 1
    );
    auto tupleForFluxSecondIterator = thrust::make_zip_iterator(tupleForFluxSecond);
    thrust::transform(
        tupleForFluxSecondIterator + localSizeY, 
        tupleForFluxSecondIterator + localSizeX * localSizeY, 
        U.begin() + localSizeY, 
        oneStepSecondFunctor()
    );

    ct.divBClean(bXOld, bYOld, U);

    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_U(U, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);

    boundary.periodicBoundaryX2nd(U);
    boundary.periodicBoundaryY2nd(U);
    MPI_Barrier(MPI_COMM_WORLD);
}


void IdealMHD2D::oneStepRK2_periodicXSymmetricY()
{
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_U(U, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
    boundary.periodicBoundaryX2nd(U);
    boundary.symmetricBoundaryY2nd(U);
    MPI_Barrier(MPI_COMM_WORLD);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bYOld.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
    thrust::copy(U.begin(), U.end(), UBar.begin());

    calculateDt();

    shiftUToCenterForCT(U);
    fluxF = fluxSolver.getFluxF(U);
    fluxG = fluxSolver.getFluxG(U);
    backUToCenterHalfForCT(U);

    auto tupleForFluxFirst = thrust::make_tuple(
        U.begin(), fluxF.begin(), fluxF.begin() - localSizeY, fluxG.begin(), fluxG.begin() - 1
    );
    auto tupleForFluxFirstIterator = thrust::make_zip_iterator(tupleForFluxFirst);
    thrust::transform(
        tupleForFluxFirstIterator + localSizeY, 
        tupleForFluxFirstIterator + localSizeX * localSizeY, 
        UBar.begin() + localSizeY, 
        oneStepFirstFunctor()
    );

    ct.setOldFlux2D(fluxF, fluxG);
    ct.divBClean(bXOld, bYOld, UBar);

    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_U(UBar, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
    boundary.periodicBoundaryX2nd(UBar);
    boundary.symmetricBoundaryY2nd(UBar);
    MPI_Barrier(MPI_COMM_WORLD);

    shiftUToCenterForCT(UBar);
    fluxF = fluxSolver.getFluxF(UBar);
    fluxG = fluxSolver.getFluxG(UBar);
    backUToCenterHalfForCT(UBar);

    auto tupleForFluxSecond = thrust::make_tuple(
        U.begin(), UBar.begin(), fluxF.begin(), fluxF.begin() - localSizeY, fluxG.begin(), fluxG.begin() - 1
    );
    auto tupleForFluxSecondIterator = thrust::make_zip_iterator(tupleForFluxSecond);
    thrust::transform(
        tupleForFluxSecondIterator + localSizeY, 
        tupleForFluxSecondIterator + localSizeX * localSizeY, 
        U.begin() + localSizeY, 
        oneStepSecondFunctor()
    );

    ct.divBClean(bXOld, bYOld, U);

    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_U(U, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);

    boundary.periodicBoundaryX2nd(U);
    boundary.symmetricBoundaryY2nd(U);
    MPI_Barrier(MPI_COMM_WORLD);
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
             + "_" + std::to_string(mPIInfo.rank)
             + ".bin";

    std::ofstream ofs(filename, std::ios::binary);
    ofs << std::fixed << std::setprecision(6);

    for (int i = mPIInfo.buffer; i < mPIInfo.localNx + mPIInfo.buffer; i++) {
        for (int j = mPIInfo.buffer; j < mPIInfo.localNy + mPIInfo.buffer; j++) {
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * mPIInfo.localSizeY].rho),  sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * mPIInfo.localSizeY].rhoU), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * mPIInfo.localSizeY].rhoV), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * mPIInfo.localSizeY].rhoW), sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * mPIInfo.localSizeY].bX),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * mPIInfo.localSizeY].bY),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * mPIInfo.localSizeY].bZ),   sizeof(double));
            ofs.write(reinterpret_cast<const char*>(&hU[j + i * mPIInfo.localSizeY].e),    sizeof(double));
        }
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
    
    double dtLocal = dt;
    double dtCommon;
    
    MPI_Allreduce(&dtLocal, &dtCommon, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    dt = dtCommon;

    cudaMemcpyToSymbol(device_dt, &dt, sizeof(double));
    cudaDeviceSynchronize();
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

    bool global_result;
    MPI_Allreduce(&result, &global_result, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

    if (dt < 0) global_result = true;

    return global_result;
}

/////////////////////

__global__ void shiftBXToCenterForCT_kernel(
    const double* tmp, 
    ConservationParameter* U, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (j < localSizeY)) {
        U[j + i * localSizeY].bX = 0.5 * (tmp[j + i * localSizeY] + tmp[j + (i - 1) * localSizeY]);
    }
}

__global__ void shiftBYToCenterForCT_kernel(
    const double* tmp, 
    ConservationParameter* U, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < localSizeX) && (0 < j) && (j < localSizeY)) {
        U[j + i * localSizeY].bY = 0.5 * (tmp[j + i * localSizeY] + tmp[j - 1 + i * localSizeY]);
    }
}


void IdealMHD2D::shiftUToCenterForCT(
    thrust::device_vector<ConservationParameter>& U
)
{
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);


    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
    
    shiftBXToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );

    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
    
    shiftBYToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
}


__global__ void backBXToCenterForCT_kernel(
    const double* tmp, 
    ConservationParameter* U, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < localSizeX - 1) && (j < localSizeY)) {
        U[j + i * localSizeY].bX = 0.5 * (tmp[j + i * localSizeY] + tmp[j + (i + 1) * localSizeY]);
    }
}

__global__ void backBYToCenterForCT_kernel(
    const double* tmp, 
    ConservationParameter* U, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < localSizeX) && (j < localSizeY - 1)) {
        U[j + i * localSizeY].bY = 0.5 * (tmp[j + i * localSizeY] + tmp[j + 1 + i * localSizeY]);
    }
}


void IdealMHD2D::backUToCenterHalfForCT(
    thrust::device_vector<ConservationParameter>& U
)
{
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);


    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
    
    backBXToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );

    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
    
    backBYToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );
}



// getter
thrust::device_vector<ConservationParameter>& IdealMHD2D::getU()
{
    return U;
}

