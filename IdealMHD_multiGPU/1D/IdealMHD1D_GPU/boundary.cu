#include "boundary.hpp"


__global__ void symmetricBoundary2nd_kernel(
    ConservationParameter* U, 
    int rank, int procs, int localSize
)
{

    if (rank == 0) {
        U[0].rho  = U[2].rho;
        U[0].rhoU = U[2].rhoU;
        U[0].rhoV = U[2].rhoV;
        U[0].rhoW = U[2].rhoW;
        U[0].bX   = U[2].bX;
        U[0].bY   = U[2].bY;
        U[0].bZ   = U[2].bZ;
        U[0].e    = U[2].e;
        U[1].rho  = U[2].rho;
        U[1].rhoU = U[2].rhoU;
        U[1].rhoV = U[2].rhoV;
        U[1].rhoW = U[2].rhoW;
        U[1].bX   = U[2].bX;
        U[1].bY   = U[2].bY;
        U[1].bZ   = U[2].bZ;
        U[1].e    = U[2].e;
    }
    
    if (rank == procs - 1) {

        U[localSize - 1].rho  = U[localSize - 3].rho;
        U[localSize - 1].rhoU = U[localSize - 3].rhoU;
        U[localSize - 1].rhoV = U[localSize - 3].rhoV;
        U[localSize - 1].rhoW = U[localSize - 3].rhoW;
        U[localSize - 1].bX   = U[localSize - 3].bX;
        U[localSize - 1].bY   = U[localSize - 3].bY;
        U[localSize - 1].bZ   = U[localSize - 3].bZ;
        U[localSize - 1].e    = U[localSize - 3].e;
        U[localSize - 2].rho  = U[localSize - 3].rho;
        U[localSize - 2].rhoU = U[localSize - 3].rhoU;
        U[localSize - 2].rhoV = U[localSize - 3].rhoV;
        U[localSize - 2].rhoW = U[localSize - 3].rhoW;
        U[localSize - 2].bX   = U[localSize - 3].bX;
        U[localSize - 2].bY   = U[localSize - 3].bY;
        U[localSize - 2].bZ   = U[localSize - 3].bZ;
        U[localSize - 2].e    = U[localSize - 3].e;
    }
}


void Boundary::symmetricBoundary2nd(
    thrust::device_vector<ConservationParameter>& U, 
    MPIInfo& mPIInfo
)
{
    int threadsPerBlock = 1;
    int blocksPerGrid = 1;

    symmetricBoundary2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.rank, mPIInfo.procs, mPIInfo.localSize
    );

    cudaDeviceSynchronize();
}

