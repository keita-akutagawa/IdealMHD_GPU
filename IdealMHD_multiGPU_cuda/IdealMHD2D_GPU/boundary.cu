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
    int localSizeX, int localSizeY, 
    MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    MPIInfo mPIInfo = *device_mPIInfo;

    if (i < localSizeX) {
        if (mPIInfo.localGridY == 0) {
            int index = 0 + i * localSizeY;

            double rho, u, v, w, bX, bY, bZ, p, e;
            ConservationParameter wallU;

            rho = U[index + mPIInfo.buffer].rho;
            u   = U[index + mPIInfo.buffer].rhoU / rho; 
            v   = U[index + mPIInfo.buffer].rhoV / rho; 
            w   = U[index + mPIInfo.buffer].rhoW / rho;
            bX  = U[index + mPIInfo.buffer].bX; 
            bY  = U[index + mPIInfo.buffer].bY;
            bZ  = U[index + mPIInfo.buffer].bZ;
            e   = U[index + mPIInfo.buffer].e;
            p   = (device_gamma_mhd - 1.0)
                * (e - 0.5 * rho * (u * u + v * v + w * w)
                - 0.5 * (bX * bX + bY * bY + bZ * bZ));
            
            wallU.rho = rho;
            wallU.rhoU = rho * u; wallU.rhoV = rho * (-v); wallU.rhoW = rho * w;
            wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
            e = p / (device_gamma_mhd - 1.0) + 0.5 * rho * (u * u + (-v) * (-v) + w * w)
            + 0.5 * (bX * bX + 0.0 * 0.0 + bZ * bZ); 
            wallU.e = e;

            for (int buf = 0; buf < mPIInfo.buffer; buf++) {
                U[index + buf] = wallU;
            }
        }
        
        if (mPIInfo.localGridY == mPIInfo.gridY - 1) {
            int index = localSizeY - 1 + i * localSizeY;

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

            for (int buf = 0; buf < mPIInfo.buffer; buf++) {
                U[index - buf] = wallU;
            }
        }
    }
}

void Boundary::wallBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    // そこまで重くないので、初期化と同じくグローバルで扱うことにする
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    wallBoundaryY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}
