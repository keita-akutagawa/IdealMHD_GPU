#include "boundary.hpp"


Boundary::Boundary(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{

    cudaMalloc(&device_mPIInfo, sizeof(MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(MPIInfo), cudaMemcpyHostToDevice);
    
}


void Boundary::periodicBoundaryX2nd(
    thrust::device_vector<ConservationParameter>& U
)
{

}


void Boundary::periodicBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    
}

///////////////////////

__global__
void symmetricBoundaryY2nd_kernel(
    ConservationParameter* U, 
    MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    MPIInfo mPIInfo = *device_mPIInfo;

    if (i < device_nx) {
        if (mPIInfo.isInside(i, 0)) {
            int index = mPIInfo.globalToLocal(i, 0);
        
            U[index    ] = U[index + 3];
            U[index + 1] = U[index + 3];
            U[index + 2] = U[index + 3];
        }
        
        if (mPIInfo.isInside(i, device_ny - 1)) {
            int index = mPIInfo.globalToLocal(i, device_ny - 1);

            U[index    ] = U[index - 3];
            U[index - 1] = U[index - 3];
            U[index - 2] = U[index - 3];
        }
    }
}

void Boundary::symmetricBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    // そこまで重くないので、初期化と同じくグローバルで扱うことにする
    int threadsPerBlock = 256;
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}
