#include "ct.hpp"


CT::CT(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      oldFluxF(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      oldFluxG(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      eZVector(mPIInfo.localSizeX * mPIInfo.localSizeY)
{
}


void CT::setOldFlux2D(
    const thrust::device_vector<Flux>& fluxF, 
    const thrust::device_vector<Flux>& fluxG
)
{
    thrust::copy(fluxF.begin(), fluxF.end(), oldFluxF.begin());
    thrust::copy(fluxG.begin(), fluxG.end(), oldFluxG.begin());
}


__global__ void getEZVector_kernel(
    const Flux* fluxF, 
    const Flux* fluxG, 
    double* eZVector, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 1 && j < localSizeY - 1) {
        double eZF1, eZF2, eZG1, eZG2, eZ;

        eZG1 = fluxG[j + i * localSizeY].f4;
        eZG2 = fluxG[j + (i + 1) * localSizeY].f4;
        eZF1 = -1.0 * fluxF[j + i * localSizeY].f5;
        eZF2 = -1.0 * fluxF[j + 1 + i * localSizeY].f5;
        eZ = 0.25 * (eZG1 + eZG2 + eZF1 + eZF2);
        eZVector[j + i * localSizeY] = eZ;
    }
}


__global__ void CT_kernel(
    const double* bXOld, const double* bYOld, 
    const double* eZVector, 
    ConservationParameter* U, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < localSizeY)) {
        U[j + i * localSizeY].bX = bXOld[j + i * localSizeY]
                                 - device_dt / device_dy * (eZVector[j + i * localSizeY] - eZVector[j - 1 + i * localSizeY]);
        U[j + i * localSizeY].bY = bYOld[j + i * localSizeY]
                                 + device_dt / device_dx * (eZVector[j + i * localSizeY] - eZVector[j + (i - 1) * localSizeY]);
    }
}


void CT::divBClean(
    const thrust::device_vector<double>& bXOld, 
    const thrust::device_vector<double>& bYOld, 
    thrust::device_vector<ConservationParameter>& U
)
{
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    getEZVector_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(oldFluxF.data()), 
        thrust::raw_pointer_cast(oldFluxG.data()), 
        thrust::raw_pointer_cast(eZVector.data()), 
        localSizeX, localSizeY
    );

    CT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()),
        thrust::raw_pointer_cast(bYOld.data()),
        thrust::raw_pointer_cast(eZVector.data()),
        thrust::raw_pointer_cast(U.data()), 
        localSizeX, localSizeY
    );

}

