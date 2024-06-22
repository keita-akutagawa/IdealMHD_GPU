#include "ct.hpp"


CT::CT()
    : oldFluxF(nx * ny), 
      oldFluxG(nx * ny), 
      eZVector(nx * ny)
{
}


void CT::setOldFlux2D(
    const thrust::device_vector<Flux>& fluxF, 
    const thrust::device_vector<Flux>& fluxG
)
{
    thrust::copy(fluxF.begin(), fluxF.end(), oldFluxF.begin());
    thrust::copy(fluxG.begin(), fluxG.end(), oldFluxG.begin());
}


__global__ void getEZVector_kernel(
    const Flux* fluxF, 
    const Flux* fluxG, 
    double* eZVector
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx - 1 && j < device_ny - 1) {
        double eZF1, eZF2, eZG1, eZG2, eZ;

        eZG1 = fluxG[j + i * device_ny].f4;
        eZG2 = fluxG[j + (i + 1) * device_ny].f4;
        eZF1 = -1.0 * fluxF[j + i * device_ny].f5;
        eZF2 = -1.0 * fluxF[j + 1 + i * device_ny].f5;
        eZ = 0.25 * (eZG1 + eZG2 + eZF1 + eZF2);
        eZVector[j + i * device_ny] = eZ;
    }
}


__global__ void CT_kernel(
    const double* bXOld, const double* bYOld, 
    const double* eZVector, 
    ConservationParameter* U
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < device_nx) && (0 < j) && (j < device_ny)) {
        U[j + i * device_ny].bX = bXOld[j + i * device_ny]
                                - device_dt / device_dy * (eZVector[j + i * device_ny] - eZVector[j - 1 + i * device_ny]);
        U[j + i * device_ny].bY = bYOld[j + i * device_ny]
                                + device_dt / device_dx * (eZVector[j + i * device_ny] - eZVector[j + (i - 1) * device_ny]);
    }
}


void CT::divBClean(
    const thrust::device_vector<double>& bXOld, 
    const thrust::device_vector<double>& bYOld, 
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    getEZVector_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(oldFluxF.data()), 
        thrust::raw_pointer_cast(oldFluxG.data()), 
        thrust::raw_pointer_cast(eZVector.data())
    );

    CT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()),
        thrust::raw_pointer_cast(bYOld.data()),
        thrust::raw_pointer_cast(eZVector.data()),
        thrust::raw_pointer_cast(U.data())
    );

}

