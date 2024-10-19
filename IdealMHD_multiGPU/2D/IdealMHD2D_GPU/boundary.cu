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
        
            U[index    ].rho  = U[index + 5].rho;
            U[index    ].rhoU = U[index + 5].rhoU;
            U[index    ].rhoV = U[index + 5].rhoV;
            U[index    ].rhoW = U[index + 5].rhoW;
            U[index    ].bX   = U[index + 5].bX;
            U[index    ].bZ   = U[index + 5].bZ;
            U[index    ].e    = U[index + 5].e;
            U[index + 1].rho  = U[index + 4].rho;
            U[index + 1].rhoU = U[index + 4].rhoU;
            U[index + 1].rhoV = U[index + 4].rhoV;
            U[index + 1].rhoW = U[index + 4].rhoW;
            U[index + 1].bX   = U[index + 4].bX;
            U[index + 1].bZ   = U[index + 4].bZ;
            U[index + 1].e    = U[index + 4].e;
            U[index + 2].rho  = U[index + 3].rho;
            U[index + 2].rhoU = U[index + 3].rhoU;
            U[index + 2].rhoV = U[index + 3].rhoV;
            U[index + 2].rhoW = U[index + 3].rhoW;
            U[index + 2].bX   = U[index + 3].bX;
            U[index + 2].bZ   = U[index + 3].bZ;
            U[index + 2].e    = U[index + 3].e;
        }
        
        if (mPIInfo.isInside(i, device_ny - 1)) {
            int index = mPIInfo.globalToLocal(i, device_ny - 1);

            U[index    ].rho  = U[index - 5].rho;
            U[index    ].rhoU = U[index - 5].rhoU;
            U[index    ].rhoV = U[index - 5].rhoV;
            U[index    ].rhoW = U[index - 5].rhoW;
            U[index    ].bX   = U[index - 5].bX;
            U[index    ].bZ   = U[index - 5].bZ;
            U[index    ].e    = U[index - 5].e;
            U[index - 1].rho  = U[index - 4].rho;
            U[index - 1].rhoU = U[index - 4].rhoU;
            U[index - 1].rhoV = U[index - 4].rhoV;
            U[index - 1].rhoW = U[index - 4].rhoW;
            U[index - 1].bX   = U[index - 4].bX;
            U[index - 1].bZ   = U[index - 4].bZ;
            U[index - 1].e    = U[index - 4].e;
            U[index - 2].rho  = U[index - 3].rho;
            U[index - 2].rhoU = U[index - 3].rhoU;
            U[index - 2].rhoV = U[index - 3].rhoV;
            U[index - 2].rhoW = U[index - 3].rhoW;
            U[index - 2].bX   = U[index - 3].bX;
            U[index - 2].bZ   = U[index - 3].bZ;
            U[index - 2].e    = U[index - 3].e;
        }
    }
}

__global__
void symmetricBoundaryY2ndBY_kernel(
    ConservationParameter* U,
    MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    MPIInfo mPIInfo = *device_mPIInfo;

    if (0 < i && i < device_nx) {
        if (mPIInfo.isInside(i, 0)) {
            int index = mPIInfo.globalToLocal(i, 0);
        
            U[index + 2].bY = U[index + 3].bY
                            + (U[index + 3].bX - U[index + 3 - device_ny].bX) / device_dx * device_dy;
            U[index + 1].bY = U[index + 2].bY
                            + (U[index + 2].bX - U[index + 2 - device_ny].bX) / device_dx * device_dy;
            U[index    ].bY = U[index + 1].bY
                            + (U[index + 1].bX - U[index + 1 - device_ny].bX) / device_dx * device_dy;
        }
        
        if (mPIInfo.isInside(i, device_ny - 1)) {
            int index = mPIInfo.globalToLocal(i, device_ny - 1);
            U[index - 3].bY = U[index - 4].bY
                            - (U[index - 3].bX - U[index - 3 - device_ny].bX) / device_dx * device_dy;
            U[index - 2].bY = U[index - 3].bY
                            - (U[index - 2].bX - U[index - 2 - device_ny].bX) / device_dx * device_dy;
            U[index - 1].bY = U[index - 2].bY
                            - (U[index - 1].bX - U[index - 1 - device_ny].bX) / device_dx * device_dy;
            U[index].bY     = U[index - 1].bY
                            - (U[index].bX - U[index - device_ny].bX) / device_dx * device_dy;
        }
    }

    if (i == 0) {
        if (mPIInfo.isInside(i, 0)) {
            int index = mPIInfo.globalToLocal(i, 0);
            U[index + 2].bY = U[index + 3].bY;
            U[index + 1].bY = U[index + 2].bY;
            U[index    ].bY = U[index + 1].bY;
        }
        
        if (mPIInfo.isInside(i, device_ny - 1)) {
            int index = mPIInfo.globalToLocal(i, device_ny - 1);
            U[index - 3].bY = U[index - 4].bY;
            U[index - 2].bY = U[index - 3].bY;
            U[index - 1].bY = U[index - 2].bY;
            U[index    ].bY = U[index - 1].bY;
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

    symmetricBoundaryY2ndBY_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}

