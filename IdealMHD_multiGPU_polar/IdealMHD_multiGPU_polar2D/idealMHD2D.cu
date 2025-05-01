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
      sourceTerm(mPIInfo.localSizeX * mPIInfo.localSizeY),
      U        (mPIInfo.localSizeX * mPIInfo.localSizeY),
      UBar     (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      tmpVector(mPIInfo.localSizeX * mPIInfo.localSizeY),
      bXOld    (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      bYOld    (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      hU       (mPIInfo.localSizeX * mPIInfo.localSizeY), 

      dtVector(mPIInfo.localNx * mPIInfo.localNy), 

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
        int index = j + i * localSizeY;

        tmp[index] = U[index].bX;
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
        int index = j + i * localSizeY;

        tmp[index] = U[index].bY;
    }
}


__global__ void oneStepFirst_kernel(
    const ConservationParameter* U, 
    const Flux* fluxF, const Flux* fluxG, 
    const SourceTerm* sourceTerm, 
    ConservationParameter* UBar, 
    int localSizeX, int localSizeY, 
    const int localGridX, const int localNx, const int buffer 
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < localSizeY)) {
        int index = j + i * localSizeY;
        double x = localNx * localGridX + (i - buffer) * device_dx + device_xmin; 

        UBar[index].rho  = U[index].rho  
                         - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f0 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f0)
                         - device_dt / device_dy / x * (fluxG[index].f0 - fluxG[index - 1].f0)
                         + device_dt * sourceTerm[index].s0;
        UBar[index].rhoU = U[index].rhoU 
                         - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f1 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f1)
                         - device_dt / device_dy / x * (fluxG[index].f1 - fluxG[index - 1].f1)
                         + device_dt * sourceTerm[index].s1;
        UBar[index].rhoV = U[index].rhoV
                         - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f2 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f2)
                         - device_dt / device_dy / x * (fluxG[index].f2 - fluxG[index - 1].f2)
                         + device_dt * sourceTerm[index].s2;
        UBar[index].rhoW = U[index].rhoW
                         - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f3 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f3)
                         - device_dt / device_dy / x * (fluxG[index].f3 - fluxG[index - 1].f3)
                         + device_dt * sourceTerm[index].s3;
        UBar[index].bX   = U[index].bX 
                         - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f4 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f4)
                         - device_dt / device_dy / x * (fluxG[index].f4 - fluxG[index - 1].f4)
                         + device_dt * sourceTerm[index].s4;
        UBar[index].bY   = U[index].bY 
                         - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f5 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f5)
                         - device_dt / device_dy / x * (fluxG[index].f5 - fluxG[index - 1].f5)
                         + device_dt * sourceTerm[index].s5;
        UBar[index].bZ   = U[index].bZ 
                         - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f6 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f6)
                         - device_dt / device_dy / x * (fluxG[index].f6 - fluxG[index - 1].f6)
                         + device_dt * sourceTerm[index].s6;
        UBar[index].e    = U[index].e 
                         - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f7 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f7)
                         - device_dt / device_dy / x * (fluxG[index].f7 - fluxG[index - 1].f7)
                         + device_dt * sourceTerm[index].s7;

    }
}


__global__ void oneStepSecond_kernel(
    const ConservationParameter* UBar, 
    const Flux* fluxF, const Flux* fluxG, 
    const SourceTerm* sourceTerm, 
    ConservationParameter* U, 
    int localSizeX, int localSizeY, 
    const int localGridX, const int localNx, const int buffer 
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < localSizeY)) {
        int index = j + i * localSizeY;
        double x = localNx * localGridX + (i - buffer) * device_dx + device_xmin; 

        U[index].rho  = 0.5 * (U[index].rho + UBar[index].rho
                      - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f0 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f0)
                      - device_dt / device_dy / x * (fluxG[index].f0 - fluxG[index - 1].f0)
                      + device_dt * sourceTerm[index].s0);
        U[index].rhoU = 0.5 * (U[index].rhoU + UBar[index].rhoU
                      - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f1 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f1)
                      - device_dt / device_dy / x * (fluxG[index].f1 - fluxG[index - 1].f1)
                      + device_dt * sourceTerm[index].s1);
        U[index].rhoV = 0.5 * (U[index].rhoV + UBar[index].rhoV
                      - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f2 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f2)
                      - device_dt / device_dy / x * (fluxG[index].f2 - fluxG[index - 1].f2)
                      + device_dt * sourceTerm[index].s2);
        U[index].rhoW = 0.5 * (U[index].rhoW + UBar[index].rhoW
                      - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f3 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f3)
                      - device_dt / device_dy / x * (fluxG[index].f3 - fluxG[index - 1].f3)
                      + device_dt * sourceTerm[index].s3);
        U[index].bX   = 0.5 * (U[index].bX + UBar[index].bX
                      - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f4 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f4)
                      - device_dt / device_dy / x * (fluxG[index].f4 - fluxG[index - 1].f4)
                      + device_dt * sourceTerm[index].s4);
        U[index].bY   = 0.5 * (U[index].bY + UBar[index].bY
                      - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f5 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f5)
                      - device_dt / device_dy / x * (fluxG[index].f5 - fluxG[index - 1].f5)
                      + device_dt * sourceTerm[index].s5);
        U[index].bZ   = 0.5 * (U[index].bZ + UBar[index].bZ
                      - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f6 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f6)
                      - device_dt / device_dy / x * (fluxG[index].f6 - fluxG[index - 1].f6)
                      + device_dt * sourceTerm[index].s6);
        U[index].e    = 0.5 * (U[index].e + UBar[index].e
                      - device_dt / device_dx / x * ((x + 0.5 * device_dx) * fluxF[index].f7 - (x - 0.5 * device_dx) * fluxF[index - localSizeY].f7)
                      - device_dt / device_dy / x * (fluxG[index].f7 - fluxG[index - 1].f7)
                      + device_dt * sourceTerm[index].s7);
    }
}


void IdealMHD2D::oneStepRK2_symmetricXperiodicY()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);


    MPI_Barrier(MPI_COMM_WORLD);
    
    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bYOld.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();

    thrust::copy(U.begin(), U.end(), UBar.begin());

    calculateDt();

    shiftUToCenterForCT(U);
    fluxF = fluxSolver.getFluxF(U);
    fluxG = fluxSolver.getFluxG(U);
    backUToCenterHalfForCT(U);

    ct.setOldFlux2D(fluxF, fluxG);

    calculateSourceTerm();

    oneStepFirst_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(sourceTerm.data()), 
        thrust::raw_pointer_cast(UBar.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.localGridX, mPIInfo.localNx, mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    boundary.wallBoundaryXLeft2nd_U(UBar);
    boundary.symmetricBoundaryXRight2nd_U(UBar);
    boundary.periodicBoundaryY2nd_U(UBar);
    
    shiftUToCenterForCT(UBar);
    fluxF = fluxSolver.getFluxF(UBar);
    fluxG = fluxSolver.getFluxG(UBar);
    backUToCenterHalfForCT(UBar);

    calculateSourceTerm();

    oneStepSecond_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(UBar.data()), 
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(sourceTerm.data()),
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.localGridX, mPIInfo.localNx, mPIInfo.buffer
    );
    cudaDeviceSynchronize();

    boundary.wallBoundaryXLeft2nd_U(U);
    boundary.symmetricBoundaryXRight2nd_U(U);
    boundary.periodicBoundaryY2nd_U(U);

    ct.divBClean(bXOld, bYOld, U);

    boundary.wallBoundaryXLeft2nd_U(U);
    boundary.symmetricBoundaryXRight2nd_U(U);
    boundary.periodicBoundaryY2nd_U(U);
}


__global__ void calculateSourceTerm_kernel(
    const Flux* fluxF, const Flux* fluxG,  
    SourceTerm* sourceTerm, 
    int localSizeX, int localSizeY, 
    const int localGridX, const int localNx, const int buffer 
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && 0 < j && j < localSizeY) {
        int index = j + i * localSizeY;
        double x = localNx * localGridX + (i - buffer) * device_dx + device_xmin; 

        sourceTerm[index].s0 = 0.0;
        sourceTerm[index].s1 = 0.5 * (fluxG[index].f2 + fluxG[index - localSizeY].f2) / x;
        sourceTerm[index].s2 = -0.5 * (fluxF[index].f2 - fluxF[index - localSizeY].f2) / x;
        sourceTerm[index].s3 = 0.0;
        sourceTerm[index].s4 = 0.0;
        sourceTerm[index].s5 = -0.5 * (fluxF[index].f5 + fluxF[index - localSizeY].f5) / x;
        sourceTerm[index].s6 = 0.0;
        sourceTerm[index].s7 = 0.0;
    }
}


void IdealMHD2D::calculateSourceTerm()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateSourceTerm_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(sourceTerm.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.localGridX, mPIInfo.localNx, mPIInfo.buffer
    );
    cudaDeviceSynchronize();
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


__global__ void calculateDtVector_kernel(
    const ConservationParameter* U, 
    double* dtVector, 
    int localNx, int localNy, int buffer, 
    int localGridX
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNx && j < localNy) {
        int localSizeY = localNy + 2 * buffer;
        int indexForU = (j + buffer) + (i + buffer) * localSizeY;
        int indexForDt = j + i * localNy;
        double x = localNx * localGridX + (i - buffer) * device_dx + device_xmin; 

        double rho, u, v, w, bX, bY, bZ, e, p, cs, ca;
        double maxSpeedX, maxSpeedY;

        rho = U[indexForU].rho;
        u   = U[indexForU].rhoU / rho;
        v   = U[indexForU].rhoV / rho;
        w   = U[indexForU].rhoW / rho;
        bX  = U[indexForU].bX;
        bY  = U[indexForU].bY;
        bZ  = U[indexForU].bZ;
        e   = U[indexForU].e;
        p   = (device_gamma_mhd - 1.0)
            * (e - 0.5 * rho * (u * u + v * v + w * w)
            - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        cs = sqrt(device_gamma_mhd * p / rho);
        ca = sqrt((bX * bX + bY * bY + bZ * bZ) / rho);

        maxSpeedX = std::abs(u) + sqrt(cs * cs + ca * ca);
        maxSpeedY = std::abs(v) + sqrt(cs * cs + ca * ca);

        dtVector[indexForDt] = 1.0 / (maxSpeedX / device_dx + maxSpeedY / x / device_dy + device_EPS);
    
    }
}


void IdealMHD2D::calculateDt()
{
    // localSizeではないので注意
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localNy + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateDtVector_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(dtVector.data()), 
        mPIInfo.localNx, mPIInfo.localNy, mPIInfo.buffer, 
        mPIInfo.localGridX
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
        int index = j + i * localSizeY;

        U[index].bX = 0.5 * (tmp[index] + tmp[index - localSizeY]);
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
        int index = j + i * localSizeY;

        U[index].bY = 0.5 * (tmp[index] + tmp[index - 1]);
    }
}


void IdealMHD2D::shiftUToCenterForCT(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);


    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    
    shiftBXToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );

    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    
    shiftBYToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
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
        int index = j + i * localSizeY;

        U[index].bX = 0.5 * (tmp[index] + tmp[index + localSizeY]);
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
        int index = j + i * localSizeY;

        U[index].bY = 0.5 * (tmp[index] + tmp[index + 1]);
    }
}


void IdealMHD2D::backUToCenterHalfForCT(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);


    copyBX_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    
    backBXToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );

    copyBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    
    backBYToCenterForCT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(tmpVector.data()), 
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
}



thrust::device_vector<ConservationParameter>& IdealMHD2D::getU()
{
    return U;
}

