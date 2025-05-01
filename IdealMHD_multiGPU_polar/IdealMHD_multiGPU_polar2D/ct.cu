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
    int localSizeX, int localSizeY, 
    int localGridX, int localNx, int buffer 
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < localSizeY)) {
        int index = j + i * localSizeY;
        double x = localNx * localGridX + (i - buffer) * device_dx + device_xmin; 

        U[index].bX = bXOld[index]
                    - device_dt / x / device_dy * (eZVector[index] - eZVector[index - 1]);
        U[index].bY = bYOld[index]
                    + device_dt / device_dx * (eZVector[index] - eZVector[index - localSizeY]);
    }
}


void CT::divBClean(
    const thrust::device_vector<double>& bXOld, 
    const thrust::device_vector<double>& bYOld, 
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    getEZVector_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(oldFluxF.data()), 
        thrust::raw_pointer_cast(oldFluxG.data()), 
        thrust::raw_pointer_cast(eZVector.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();

    CT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()),
        thrust::raw_pointer_cast(bYOld.data()),
        thrust::raw_pointer_cast(eZVector.data()),
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.localGridX, mPIInfo.localNx, mPIInfo.buffer
    );
    cudaDeviceSynchronize();
}

