#include "boundary.hpp"
#include "const.hpp"


__global__
void periodicBoundary2nd_kernel(ConservationParameter* U)
{
    U[0].rho  = U[nx-4].rho;
    U[0].rhoU = U[nx-4].rhoU;
    U[0].rhoV = U[nx-4].rhoV;
    U[0].rhoW = U[nx-4].rhoW;
    U[0].bX   = U[nx-4].bX;
    U[0].bY   = U[nx-4].bY;
    U[0].bZ   = U[nx-4].bZ;
    U[0].e    = U[nx-4].e;
    U[1].rho  = U[nx-3].rho;
    U[1].rhoU = U[nx-3].rhoU;
    U[1].rhoV = U[nx-3].rhoV;
    U[1].rhoW = U[nx-3].rhoW;
    U[1].bX   = U[nx-3].bX;
    U[1].bY   = U[nx-3].bY;
    U[1].bZ   = U[nx-3].bZ;
    U[1].e    = U[nx-3].e;

    U[nx-2].rho  = U[2].rho;
    U[nx-2].rhoU = U[2].rhoU;
    U[nx-2].rhoV = U[2].rhoV;
    U[nx-2].rhoW = U[2].rhoW;
    U[nx-2].bX   = U[2].bX;
    U[nx-2].bY   = U[2].bY;
    U[nx-2].bZ   = U[2].bZ;
    U[nx-2].e    = U[2].e;
    U[nx-1].rho  = U[3].rho;
    U[nx-1].rhoU = U[3].rhoU;
    U[nx-1].rhoV = U[3].rhoV;
    U[nx-1].rhoW = U[3].rhoW;
    U[nx-1].bX   = U[3].bX;
    U[nx-1].bY   = U[3].bY;
    U[nx-1].bZ   = U[3].bZ;
    U[nx-1].e    = U[3].e;
}

void Boundary::periodicBoundary(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 1;
    int blocksPerGrid = 1;

    periodicBoundary2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}


__global__
void symmetricBoundary2nd_kernel(ConservationParameter* U)
{
    
}


void Boundary::symmetricBoundary2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    int threadsPerBlock = 1;
    int blocksPerGrid = 1;

    symmetricBoundary2nd_kernel<<<blocksPerGrid, threadsPerBlock>>>(thrust::raw_pointer_cast(U.data()));

    cudaDeviceSynchronize();
}

