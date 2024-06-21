#include "ct.hpp"


CT::CT()
    : EzVector(nx * ny), 
      oldFluxF(nx * ny), 
      oldFluxG(nx * ny)
{
}


void CT::setOldFlux2D(
    const thrust::device_vector<Flux>& fluxF, 
    const thrust::device_vector<Flux>& fluxG
)
{
    thrust::copy(oldFluxF.begin(), oldFluxF.end(), fluxF.begin());
    thrust::copy(oldFluxG.begin(), oldFluxG.end(), fluxG.begin());
}


__global__ void divBClean_kernel(
    const Flux* fluxF, 
    const Flux* fluxG, 
    double* EzVector
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx - 1 && j < device_ny - 1) {
        double ezF1, ezF2, ezG1, ezG2, ez;

        ezG1 = fluxG[j + i * device_ny].f4;
        ezG2 = fluxG[j + (i+1) * device_ny].f4;
        ezF1 = -1.0 * fluxF[j + i * device_ny].f5;
        ezF2 = -1.0 * fluxF[j + 1 + i * device_ny].f5;
        ez = 0.25 * (ezG1 + ezG2 + ezF1 + ezF2);
        EzVector[j + i * device_ny] = ez;
    }
}


void CT::divBClean(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    divBClean_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(oldFluxF.data()), 
        thrust::raw_pointer_cast(oldFluxG.data()), 
        thrust::raw_pointer_cast(EzVector.data())
    );


    for (int i = 1; i < nx; i++) {
        for (int j = 1; j < ny; j++) {
            U[4][i][j] = bxOld[i][j]
                       - dt / dy * (EzVector[i][j] - EzVector[i][j-1]);
            U[5][i][j] = byOld[i][j]
                       + dt / dx * (EzVector[i][j] - EzVector[i-1][j]);
        }
    }

}

