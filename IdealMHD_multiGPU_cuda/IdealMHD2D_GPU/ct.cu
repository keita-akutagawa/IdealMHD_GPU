#include "ct.hpp"


CT::CT(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 

      oldNumericalFluxF_f5(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      oldNumericalFluxG_f4(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      oldFluxF_f5         (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      oldFluxG_f4         (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      oldNumericalFluxF_f0(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      oldNumericalFluxG_f0(mPIInfo.localSizeX * mPIInfo.localSizeY), 

      nowNumericalFluxF_f5(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      nowNumericalFluxG_f4(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      nowFluxF_f5         (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      nowFluxG_f4         (mPIInfo.localSizeX * mPIInfo.localSizeY), 
      nowNumericalFluxF_f0(mPIInfo.localSizeX * mPIInfo.localSizeY), 
      nowNumericalFluxG_f0(mPIInfo.localSizeX * mPIInfo.localSizeY), 

      eZVector            (mPIInfo.localSizeX * mPIInfo.localSizeY)
{
}


__global__ void setFlux_kernel(
    const Flux* fluxF, const Flux* fluxG, 
    const ConservationParameter* U, 
    double* NumericalFluxF_f5, double* NumericalFluxG_f4, 
    double* FluxF_f5, double* FluxG_f4, 
    double* NumericalFluxF_f0, double* NumericalFluxG_f0, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (0 < i && i < localSizeX - 1 && 0 < j && j < localSizeY - 1) {
        int index = j + i * localSizeY;
        double rho, u, v, bX, bY;
        double xPosition = i * device_dx, yPosition = j * device_dy;
        double jZ;
        double eta;

        rho = U[index].rho;
        u   = U[index].rhoU / rho;
        v   = U[index].rhoV / rho;
        bX  = 0.5 * (U[index].bX + U[index - localSizeY].bX);
        bY  = 0.5 * (U[index].bY + U[index - 1].bY);
        jZ = 0.25 * (
            (U[index + localSizeY].bY - U[index].bY) / device_dx - (U[index + 1].bX - U[index].bX) / device_dy //右上
          + (U[index - 1 + localSizeY].bY - U[index - 1].bY) / device_dx - (U[index].bX - U[index - 1].bX) / device_dy //右下
          + (U[index - 1].bY - U[index - 1 - localSizeY].bY) / device_dx - (U[index - localSizeY].bX - U[index - localSizeY - 1].bX) / device_dy //左下
          + (U[index].bY - U[index - localSizeY].bY) / device_dx - (U[index + 1 - localSizeY].bX - U[index - localSizeY].bX) / device_dy //左上
        );
        eta = getEta(xPosition, yPosition);
  
        NumericalFluxF_f5[index] = fluxF[index].f5;
        NumericalFluxG_f4[index] = fluxG[index].f4;
        FluxF_f5[index] = u * bY - v * bX - eta * jZ;
        FluxG_f4[index] = -(u * bY - v * bX - eta * jZ);
        NumericalFluxF_f0[index] = fluxF[index].f0;
        NumericalFluxG_f0[index] = fluxG[index].f0;
    }
}


void CT::setOldFlux2D(
    const thrust::device_vector<Flux>& fluxF, 
    const thrust::device_vector<Flux>& fluxG, 
    const thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    setFlux_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(oldNumericalFluxF_f5.data()), 
        thrust::raw_pointer_cast(oldNumericalFluxG_f4.data()), 
        thrust::raw_pointer_cast(oldFluxF_f5.data()), 
        thrust::raw_pointer_cast(oldFluxG_f4.data()), 
        thrust::raw_pointer_cast(oldNumericalFluxF_f0.data()), 
        thrust::raw_pointer_cast(oldNumericalFluxG_f0.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
}


void CT::setNowFlux2D(
    const thrust::device_vector<Flux>& fluxF, 
    const thrust::device_vector<Flux>& fluxG, 
    const thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    setFlux_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(fluxF.data()), 
        thrust::raw_pointer_cast(fluxG.data()), 
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(nowNumericalFluxF_f5.data()), 
        thrust::raw_pointer_cast(nowNumericalFluxG_f4.data()), 
        thrust::raw_pointer_cast(nowFluxF_f5.data()), 
        thrust::raw_pointer_cast(nowFluxG_f4.data()), 
        thrust::raw_pointer_cast(nowNumericalFluxF_f0.data()), 
        thrust::raw_pointer_cast(nowNumericalFluxG_f0.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
}


__global__ void getEZVector_kernel(
    const double* oldNumericalFluxF_f5, 
    const double* oldNumericalFluxG_f4, 
    const double* oldFluxF_f5, 
    const double* oldFluxG_f4, 
    const double* oldNumericalFluxF_f0, 
    const double* oldNumericalFluxG_f0, 
    const double* nowNumericalFluxF_f5, 
    const double* nowNumericalFluxG_f4, 
    const double* nowFluxF_f5, 
    const double* nowFluxG_f4, 
    const double* nowNumericalFluxF_f0, 
    const double* nowNumericalFluxG_f0, 
    double* eZVector, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 1 && j < localSizeY - 1) {
        double eZOld_arithmeticAverage, eZOld_S, eZOld_N, eZOld_W, eZOld_E;
        double eZOld;
        double eZNow_arithmeticAverage, eZNow_S, eZNow_N, eZNow_W, eZNow_E;
        double eZNow;
        int index = j + i * localSizeY;

        eZOld_arithmeticAverage = 0.25 * (
            - oldNumericalFluxF_f5[index] - oldNumericalFluxF_f5[index + 1]
            + oldNumericalFluxG_f4[index] + oldNumericalFluxG_f4[index + localSizeY]
        );

        eZOld_S = (1.0 + sign(oldNumericalFluxF_f0[index])) 
             * (oldNumericalFluxG_f4[index] - oldFluxG_f4[index])
             + (1.0 - sign(oldNumericalFluxF_f0[index])) 
             * (oldNumericalFluxG_f4[index + localSizeY] - oldFluxG_f4[index + localSizeY]);
        eZOld_N = (1.0 + sign(oldNumericalFluxF_f0[index + 1])) 
             * (oldFluxG_f4[index + 1] - oldNumericalFluxG_f4[index])
             + (1.0 - sign(oldNumericalFluxF_f0[index + 1])) 
             * (oldFluxG_f4[index + 1 + localSizeY] - oldNumericalFluxG_f4[index + localSizeY]);
        eZOld_W = -(1.0 + sign(oldNumericalFluxG_f0[index])) 
             * (oldNumericalFluxF_f5[index] - oldFluxF_f5[index])
             -(1.0 - sign(oldNumericalFluxG_f0[index])) 
             * (oldNumericalFluxF_f5[index + 1] - oldFluxF_f5[index + 1]);
        eZOld_E = -(1.0 + sign(oldNumericalFluxG_f0[index + localSizeY])) 
             * (oldFluxF_f5[index + localSizeY] - oldNumericalFluxF_f5[index])
             -(1.0 - sign(oldNumericalFluxG_f0[index + localSizeY])) 
             * (oldFluxF_f5[index + 1 + localSizeY] - oldNumericalFluxF_f5[index + 1]);

        eZOld = eZOld_arithmeticAverage + 0.125 * (eZOld_S - eZOld_N + eZOld_W - eZOld_E);

        eZNow_arithmeticAverage = 0.25 * (
            - nowNumericalFluxF_f5[index] - nowNumericalFluxF_f5[index + 1]
            + nowNumericalFluxG_f4[index] + nowNumericalFluxG_f4[index + localSizeY]
        );

        eZNow_S = (1.0 + sign(nowNumericalFluxF_f0[index])) 
             * (nowNumericalFluxG_f4[index] - nowFluxG_f4[index])
             + (1.0 - sign(nowNumericalFluxF_f0[index])) 
             * (nowNumericalFluxG_f4[index + localSizeY] - nowFluxG_f4[index + localSizeY]);
        eZNow_N = (1.0 + sign(nowNumericalFluxF_f0[index + 1])) 
             * (nowFluxG_f4[index + 1] - nowNumericalFluxG_f4[index])
             + (1.0 - sign(nowNumericalFluxF_f0[index + 1])) 
             * (nowFluxG_f4[index + 1 + localSizeY] - nowNumericalFluxG_f4[index + localSizeY]);
        eZNow_W = -(1.0 + sign(nowNumericalFluxG_f0[index])) 
             * (nowNumericalFluxF_f5[index] - nowFluxF_f5[index])
             -(1.0 - sign(nowNumericalFluxG_f0[index])) 
             * (nowNumericalFluxF_f5[index + 1] - nowFluxF_f5[index + 1]);
        eZNow_E = -(1.0 + sign(nowNumericalFluxG_f0[index + localSizeY])) 
             * (nowFluxF_f5[index + localSizeY] - nowNumericalFluxF_f5[index])
             -(1.0 - sign(nowNumericalFluxG_f0[index + localSizeY])) 
             * (nowFluxF_f5[index + 1 + localSizeY] - nowNumericalFluxF_f5[index + 1]);

        eZNow = eZNow_arithmeticAverage + 0.125 * (eZNow_S - eZNow_N + eZNow_W - eZNow_E);

        eZVector[index] = 0.5 * (eZOld + eZNow);
    }
}


__global__ void CT_kernel(
    const double* bXOld, const double* bYOld, 
    const double* eZVector, 
    ConservationParameter* U, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX) && (0 < j) && (j < localSizeY)) {
        int index = j + i * localSizeY;

        U[index].bX = bXOld[index]
                    - device_dt / device_dy * (eZVector[index] - eZVector[index - 1]);
        U[index].bY = bYOld[index]
                    + device_dt / device_dx * (eZVector[index] - eZVector[index - localSizeY]);
    }
}


void CT::divBClean(
    const thrust::device_vector<double>& bXOld, 
    const thrust::device_vector<double>& bYOld, 
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    getEZVector_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(oldNumericalFluxF_f5.data()), 
        thrust::raw_pointer_cast(oldNumericalFluxG_f4.data()), 
        thrust::raw_pointer_cast(oldFluxF_f5.data()), 
        thrust::raw_pointer_cast(oldFluxG_f4.data()), 
        thrust::raw_pointer_cast(oldNumericalFluxF_f0.data()), 
        thrust::raw_pointer_cast(oldNumericalFluxG_f0.data()), 
        thrust::raw_pointer_cast(nowNumericalFluxF_f5.data()), 
        thrust::raw_pointer_cast(nowNumericalFluxG_f4.data()), 
        thrust::raw_pointer_cast(nowFluxF_f5.data()), 
        thrust::raw_pointer_cast(nowFluxG_f4.data()), 
        thrust::raw_pointer_cast(nowNumericalFluxF_f0.data()), 
        thrust::raw_pointer_cast(nowNumericalFluxG_f0.data()), 
        thrust::raw_pointer_cast(eZVector.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();

    CT_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(bXOld.data()),
        thrust::raw_pointer_cast(bYOld.data()),
        thrust::raw_pointer_cast(eZVector.data()),
        thrust::raw_pointer_cast(U.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
    cudaDeviceSynchronize();
}

