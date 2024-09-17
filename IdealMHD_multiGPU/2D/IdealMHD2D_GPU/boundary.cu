#include "boundary.hpp"


Boundary::Boundary(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
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

//KHのテストのときに復活させる

/*
__global__
void symmetricBoundaryX2nd_kernel(
    ConservationParameter* U, 
    int rank, int procs, int localSizeX, int localSizeY
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank == )
    U[j + 0 * device_ny].rho  = U[j + 2 * device_ny].rho;
    U[j + 0 * device_ny].rhoU = U[j + 2 * device_ny].rhoU;
    U[j + 0 * device_ny].rhoV = U[j + 2 * device_ny].rhoV;
    U[j + 0 * device_ny].rhoW = U[j + 2 * device_ny].rhoW;
    U[j + 0 * device_ny].bX   = U[j + 2 * device_ny].bX;
    U[j + 0 * device_ny].bY   = U[j + 2 * device_ny].bY;
    U[j + 0 * device_ny].bZ   = U[j + 2 * device_ny].bZ;
    U[j + 0 * device_ny].e    = U[j + 2 * device_ny].e;
    U[j + 1 * device_ny].rho  = U[j + 2 * device_ny].rho;
    U[j + 1 * device_ny].rhoU = U[j + 2 * device_ny].rhoU;
    U[j + 1 * device_ny].rhoV = U[j + 2 * device_ny].rhoV;
    U[j + 1 * device_ny].rhoW = U[j + 2 * device_ny].rhoW;
    U[j + 1 * device_ny].bX   = U[j + 2 * device_ny].bX;
    U[j + 1 * device_ny].bY   = U[j + 2 * device_ny].bY;
    U[j + 1 * device_ny].bZ   = U[j + 2 * device_ny].bZ;
    U[j + 1 * device_ny].e    = U[j + 2 * device_ny].e;

    U[j + (device_nx-1) * device_ny].rho  = U[j + (device_nx-3) * device_ny].rho;
    U[j + (device_nx-1) * device_ny].rhoU = U[j + (device_nx-3) * device_ny].rhoU;
    U[j + (device_nx-1) * device_ny].rhoV = U[j + (device_nx-3) * device_ny].rhoV;
    U[j + (device_nx-1) * device_ny].rhoW = U[j + (device_nx-3) * device_ny].rhoW;
    U[j + (device_nx-1) * device_ny].bX   = U[j + (device_nx-3) * device_ny].bX;
    U[j + (device_nx-1) * device_ny].bY   = U[j + (device_nx-3) * device_ny].bY;
    U[j + (device_nx-1) * device_ny].bZ   = U[j + (device_nx-3) * device_ny].bZ;
    U[j + (device_nx-1) * device_ny].e    = U[j + (device_nx-3) * device_ny].e;
    U[j + (device_nx-2) * device_ny].rho  = U[j + (device_nx-3) * device_ny].rho;
    U[j + (device_nx-2) * device_ny].rhoU = U[j + (device_nx-3) * device_ny].rhoU;
    U[j + (device_nx-2) * device_ny].rhoV = U[j + (device_nx-3) * device_ny].rhoV;
    U[j + (device_nx-2) * device_ny].rhoW = U[j + (device_nx-3) * device_ny].rhoW;
    U[j + (device_nx-2) * device_ny].bX   = U[j + (device_nx-3) * device_ny].bX;
    U[j + (device_nx-2) * device_ny].bY   = U[j + (device_nx-3) * device_ny].bY;
    U[j + (device_nx-2) * device_ny].bZ   = U[j + (device_nx-3) * device_ny].bZ;
    U[j + (device_nx-2) * device_ny].e    = U[j + (device_nx-3) * device_ny].e;
}


void Boundary::symmetricBoundaryX2nd(
    thrust::device_vector<ConservationParameter>& U, 
    MPIInfo& mPIInfo
)
{
    int threadsPerBlock = mPIInfo.localSizeY;
    int blocksPerGrid = 1;

    symmetricBoundaryX2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.rank, mPIInfo.procs, mPIInfo.localSizeX, mPIInfo.localSizeY
    );

    cudaDeviceSynchronize();
}


__global__
void symmetricBoundaryY2nd_kernel(
    ConservationParameter* U, 
    int rank, int procs, int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    U[0 + i * device_ny].rho  = U[2 + i * device_ny].rho;
    U[0 + i * device_ny].rhoU = U[2 + i * device_ny].rhoU;
    U[0 + i * device_ny].rhoV = U[2 + i * device_ny].rhoV;
    U[0 + i * device_ny].rhoW = U[2 + i * device_ny].rhoW;
    U[0 + i * device_ny].bX   = U[2 + i * device_ny].bX;
    U[0 + i * device_ny].bY   = U[2 + i * device_ny].bY;
    U[0 + i * device_ny].bZ   = U[2 + i * device_ny].bZ;
    U[0 + i * device_ny].e    = U[2 + i * device_ny].e;
    U[1 + i * device_ny].rho  = U[2 + i * device_ny].rho;
    U[1 + i * device_ny].rhoU = U[2 + i * device_ny].rhoU;
    U[1 + i * device_ny].rhoV = U[2 + i * device_ny].rhoV;
    U[1 + i * device_ny].rhoW = U[2 + i * device_ny].rhoW;
    U[1 + i * device_ny].bX   = U[2 + i * device_ny].bX;
    U[1 + i * device_ny].bY   = U[2 + i * device_ny].bY;
    U[1 + i * device_ny].bZ   = U[2 + i * device_ny].bZ;
    U[1 + i * device_ny].e    = U[2 + i * device_ny].e;

    U[device_nx-1 + i * device_ny].rho  = U[device_nx-3 + i * device_ny].rho;
    U[device_nx-1 + i * device_ny].rhoU = U[device_nx-3 + i * device_ny].rhoU;
    U[device_nx-1 + i * device_ny].rhoV = U[device_nx-3 + i * device_ny].rhoV;
    U[device_nx-1 + i * device_ny].rhoW = U[device_nx-3 + i * device_ny].rhoW;
    U[device_nx-1 + i * device_ny].bX   = U[device_nx-3 + i * device_ny].bX;
    U[device_nx-1 + i * device_ny].bY   = U[device_nx-3 + i * device_ny].bY;
    U[device_nx-1 + i * device_ny].bZ   = U[device_nx-3 + i * device_ny].bZ;
    U[device_nx-1 + i * device_ny].e    = U[device_nx-3 + i * device_ny].e;
    U[device_nx-2 + i * device_ny].rho  = U[device_nx-3 + i * device_ny].rho;
    U[device_nx-2 + i * device_ny].rhoU = U[device_nx-3 + i * device_ny].rhoU;
    U[device_nx-2 + i * device_ny].rhoV = U[device_nx-3 + i * device_ny].rhoV;
    U[device_nx-2 + i * device_ny].rhoW = U[device_nx-3 + i * device_ny].rhoW;
    U[device_nx-2 + i * device_ny].bX   = U[device_nx-3 + i * device_ny].bX;
    U[device_nx-2 + i * device_ny].bY   = U[device_nx-3 + i * device_ny].bY;
    U[device_nx-2 + i * device_ny].bZ   = U[device_nx-3 + i * device_ny].bZ;
    U[device_nx-2 + i * device_ny].e    = U[device_nx-3 + i * device_ny].e;
}


void Boundary::symmetricBoundaryY2nd(
    thrust::device_vector<ConservationParameter>& U, 
    MPIInfo& mPIInfo
)
{
    int threadsPerBlock = mPIInfo.localSizeX;
    int blocksPerGrid = 1;

    symmetricBoundaryY2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.rank, mPIInfo.procs, mPIInfo.localSizeX, mPIInfo.localSizeY
    );

    cudaDeviceSynchronize();
}
*/

