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
    sendrecv_U_x(U, mPIInfo);
}


void Boundary::periodicBoundaryY2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    sendrecv_U_y(U, mPIInfo);
}

///////////////////////

__global__
void wallBoundaryY2nd_U_kernel(
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
            wallU.rhoU = rho * 0.0; wallU.rhoV = rho * 0.0; wallU.rhoW = rho * 0.0;
            wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
            e = p / (device_gamma_mhd - 1.0) + 0.5 * rho * (0.0 * 0.0 + 0.0 * 0.0 + 0.0 * 0.0)
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

            rho = U[index - mPIInfo.buffer].rho;
            u   = U[index - mPIInfo.buffer].rhoU / rho; 
            v   = U[index - mPIInfo.buffer].rhoV / rho; 
            w   = U[index - mPIInfo.buffer].rhoW / rho;
            bX  = U[index - mPIInfo.buffer].bX; 
            bY  = U[index - mPIInfo.buffer].bY;
            bZ  = U[index - mPIInfo.buffer].bZ;
            e   = U[index - mPIInfo.buffer].e;
            p   = (device_gamma_mhd - 1.0)
                * (e - 0.5 * rho * (u * u + v * v + w * w)
                - 0.5 * (bX * bX + bY * bY + bZ * bZ));
            
            wallU.rho = rho;
            wallU.rhoU = rho * 0.0; wallU.rhoV = rho * 0.0; wallU.rhoW = rho * 0.0;
            wallU.bX = bX; wallU.bY = 0.0; wallU.bZ = bZ;
            e = p / (device_gamma_mhd - 1.0) + 0.5 * rho * (0.0 * 0.0 + 0.0 * 0.0 + 0.0 * 0.0)
            + 0.5 * (bX * bX + 0.0 * 0.0 + bZ * bZ); 
            wallU.e = e;

            for (int buf = 0; buf < mPIInfo.buffer; buf++) {
                U[index - buf] = wallU;
            }
        }
    }
}

void Boundary::wallBoundaryY2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    // そこまで重くないので、初期化と同じくグローバルで扱うことにする
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    wallBoundaryY2nd_U_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}

//////////

__global__
void symmetricBoundaryX2nd_U_kernel(
    ConservationParameter* U, 
    int localSizeX, int localSizeY, 
    MPIInfo* device_mPIInfo
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    MPIInfo mPIInfo = *device_mPIInfo;

    if (j < localSizeY) {
        if (mPIInfo.localGridX == 0) {
            int index = j + 0 * localSizeY;

            for (int buf = 0; buf < mPIInfo.buffer; buf++) {
                U[index + buf * localSizeY] = U[index + mPIInfo.buffer * localSizeY];
            }
        }
        
        if (mPIInfo.localGridX == mPIInfo.gridX - 1) {
            int index = j + (localSizeX - 1) * localSizeY;

            for (int buf = 0; buf < mPIInfo.buffer; buf++) {
                U[index - buf * localSizeY] = U[index - mPIInfo.buffer * localSizeY];
            }
        }
    }
}

void Boundary::symmetricBoundaryX2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeY + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryX2nd_U_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}


__global__
void symmetricBoundaryY2nd_U_kernel(
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

            for (int buf = 0; buf < mPIInfo.buffer; buf++) {
                U[index + buf] = U[index + mPIInfo.buffer];
            }
        }
        
        if (mPIInfo.localGridY == mPIInfo.gridY - 1) {
            int index = localSizeY - 1 + i * localSizeY;

            for (int buf = 0; buf < mPIInfo.buffer; buf++) {
                U[index - buf] = U[index - mPIInfo.buffer];
            }
        }
    }
}

void Boundary::symmetricBoundaryY2nd_U(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (mPIInfo.localSizeX + threadsPerBlock - 1) / threadsPerBlock;

    symmetricBoundaryY2nd_U_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
}


