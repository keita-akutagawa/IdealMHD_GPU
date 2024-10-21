#include "boundary.hpp"


Boundary::Boundary(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{

    cudaMalloc(&device_mPIInfo, sizeof(MPIInfo));
    cudaMemcpy(device_mPIInfo, &mPIInfo, sizeof(MPIInfo), cudaMemcpyHostToDevice);
    
}


void Boundary::periodicBoundaryX2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{

}


void Boundary::periodicBoundaryY2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    
}

void Boundary::periodicBoundaryX2nd_flux(
    thrust::device_vector<Flux>& fluxF, 
    thrust::device_vector<Flux>& fluxG
)
{

}


void Boundary::periodicBoundaryY2nd_flux(
    thrust::device_vector<Flux>& fluxF, 
    thrust::device_vector<Flux>& fluxG
)
{
    
}

///////////////////////

__global__
void symmetricBoundaryY2nd_U_kernel(
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

void Boundary::symmetricBoundaryY2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    // そこまで重くないので、初期化と同じくグローバルで扱うことにする
    int threadsPerBlock = 256;
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryY2nd_U_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}


__global__
void symmetricBoundaryY2nd_flux_kernel(
    Flux* fluxF, Flux* fluxG, 
    MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    MPIInfo mPIInfo = *device_mPIInfo;

    if (i < device_nx) {
        if (mPIInfo.isInside(i, 0)) {
            int index = mPIInfo.globalToLocal(i, 0);
        
            fluxF[index    ] = fluxF[index + 3];
            fluxF[index + 1] = fluxF[index + 3];
            fluxF[index + 2] = fluxF[index + 3];
            fluxG[index    ] = fluxG[index + 3];
            fluxG[index + 1] = fluxG[index + 3];
            fluxG[index + 2] = fluxG[index + 3];
        }
        
        if (mPIInfo.isInside(i, device_ny - 1)) {
            int index = mPIInfo.globalToLocal(i, device_ny - 1);

            fluxF[index    ] = fluxF[index - 3];
            fluxF[index - 1] = fluxF[index - 3];
            fluxF[index - 2] = fluxF[index - 3];
            fluxG[index    ] = fluxG[index - 3];
            fluxG[index - 1] = fluxG[index - 3];
            fluxG[index - 2] = fluxG[index - 3];
        }
    }
}

void Boundary::symmetricBoundaryY2nd_flux(
    thrust::device_vector<Flux>& fluxF, 
    thrust::device_vector<Flux>& fluxG
)
{
    // そこまで重くないので、初期化と同じくグローバルで扱うことにする
    int threadsPerBlock = 256;
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryY2nd_flux_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()),  
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}

