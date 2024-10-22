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
void wallBoundaryY2nd_kernel(
    ConservationParameter* U, 
    MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    MPIInfo mPIInfo = *device_mPIInfo;

    if (i < device_nx) {
        if (mPIInfo.isInside(i, 0)) {
            int index = mPIInfo.globalToLocal(i, 0);

            double rho, u, v, w, bX, bY, bZ, p, e;
            ConservationParameter wallU;

            rho = U[index + 3].rho;
            u   = U[index + 3].rhoU / rho; 
            v   = U[index + 3].rhoV / rho; 
            w   = U[index + 3].rhoW / rho;
            bX  = U[index + 3].bX; 
            bY  = U[index + 3].bY;
            bZ  = U[index + 3].bZ;
            e   = U[index + 3].e;
            p   = (device_gamma_mhd - 1.0)
                * (e - 0.5 * rho * (u * u + v * v + w * w)
                - 0.5 * (bX * bX + bY * bY + bZ * bZ));
            
            wallU.rho = rho;
            wallU.rhoU = rho * u; wallU.rhoV = rho * (-v); wallU.rhoW = rho * w;
            wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
            e = p / (device_gamma_mhd - 1.0) + 0.5 * rho * (u * u + (-v) * (-v) + w * w)
            + 0.5 * (bX * bX + 0.0 * 0.0 + bZ * bZ); 
            wallU.e = e;
        
            U[index    ] = wallU;
            U[index + 1] = wallU;
            U[index + 2] = wallU;
        }
        
        if (mPIInfo.isInside(i, device_ny - 1)) {
            int index = mPIInfo.globalToLocal(i, device_ny - 1);

            double rho, u, v, w, bX, bY, bZ, p, e;
            ConservationParameter wallU;

            rho = U[index - 3].rho;
            u   = U[index - 3].rhoU / rho; 
            v   = U[index - 3].rhoV / rho; 
            w   = U[index - 3].rhoW / rho;
            bX  = U[index - 3].bX; 
            bY  = U[index - 3].bY;
            bZ  = U[index - 3].bZ;
            e   = U[index - 3].e;
            p   = (device_gamma_mhd - 1.0)
                * (e - 0.5 * rho * (u * u + v * v + w * w)
                - 0.5 * (bX * bX + bY * bY + bZ * bZ));
            
            wallU.rho = rho;
            wallU.rhoU = rho * u; wallU.rhoV = rho * (-v); wallU.rhoW = rho * w;
            wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
            e = p / (device_gamma_mhd - 1.0) + 0.5 * rho * (u * u + (-v) * (-v) + w * w)
            + 0.5 * (bX * bX + 0.0 * 0.0 + bZ * bZ); 
            wallU.e = e;

            U[index    ] = wallU;
            U[index - 1] = wallU;
            U[index - 2] = wallU;
        }
    }
}

void Boundary::wallBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    // そこまで重くないので、初期化と同じくグローバルで扱うことにする
    int threadsPerBlock = 256;
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

    wallBoundaryY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}
