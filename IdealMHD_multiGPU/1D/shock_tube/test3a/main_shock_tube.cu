#include "main_shock_tube_const.hpp"


__global__ void initializeU_kernel(
    ConservationParameter* U, 
    double rhoL0, double uL0, double vL0, double wL0, double bXL0, double bYL0, double bZL0, double pL0, double eL0, 
    double rhoR0, double uR0, double vR0, double wR0, double bXR0, double bYR0, double bZR0, double pR0, double eR0, 
    MPIInfo* device_mPIInfo
) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < device_nx) {
        MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i)) {
            int index = mPIInfo.globalToLocal(i);

            if (i < device_nx / 2) {
                U[index].rho  = rhoL0;
                U[index].rhoU = rhoL0 * uL0;
                U[index].rhoV = rhoL0 * vL0;
                U[index].rhoW = rhoL0 * wL0;
                U[index].bX   = bXL0;
                U[index].bY   = bYL0;
                U[index].bZ   = bZL0;
                U[index].e    = eL0;
            } else {
                U[index].rho  = rhoR0;
                U[index].rhoU = rhoR0 * uR0;
                U[index].rhoV = rhoR0 * vR0;
                U[index].rhoW = rhoR0 * wR0;
                U[index].bX   = bXR0;
                U[index].bY   = bYR0;
                U[index].bZ   = bZR0;
                U[index].e    = eR0;
            }
        }
    }
}

void IdealMHD1D::initializeU()
{
    double rhoL0, uL0, vL0, wL0, bXL0, bYL0, bZL0, pL0, eL0;
    double rhoR0, uR0, vR0, wR0, bXR0, bYR0, bZR0, pR0, eR0;

    rhoL0 = 0.1;
    uL0 = 50.0; vL0 = 0.0; wL0 = 0.0;
    bXL0 = 0.0; bYL0 = -1.0 / sqrt(4.0 * PI); bZL0 = -2.0 / sqrt(4.0 * PI);
    pL0 = 0.4;
    eL0 = pL0 / (gamma_mhd - 1.0)
        + 0.5 * rhoL0 * (uL0 * uL0 + vL0 * vL0 + wL0 * wL0)
        + 0.5 * (bXL0 * bXL0 + bYL0 * bYL0 + bZL0 * bZL0);

    rhoR0 = 0.1;
    uR0 = 0.0; vR0 = 0.0; wR0 = 0.0;
    bXR0 = 0.0; bYR0 = 1.0 / sqrt(4.0 * PI); bZR0 = 2.0 / sqrt(4.0 * PI);
    pR0 = 0.2;
    eR0 = pR0 / (gamma_mhd - 1.0)
        + 0.5 * rhoR0 * (uR0 * uR0 + vR0 * vR0 + wR0 * wR0)
        + 0.5 * (bXR0 * bXR0 + bYR0 * bYR0 + bZR0 * bZR0);
    

    int threadsPerBlock = 256;
    int blocksPerGrid = (nx + threadsPerBlock - 1) / threadsPerBlock;

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        rhoL0, uL0, vL0, wL0, bXL0, bYL0, bZL0, pL0, eL0, 
        rhoR0, uR0, vR0, wR0, bXR0, bYR0, bZR0, pR0, eR0, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_U(U, mPIInfo);
    boundary.symmetricBoundary2nd(U, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPIInfo mPIInfo;
    setupInfo(mPIInfo, buffer);

    cudaSetDevice(mPIInfo.rank);

    initializeDeviceConstants();


    IdealMHD1D idealMHD1D(mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);

    idealMHD1D.initializeU();

    for (int step = 0; step < totalStep + 1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (step % recordStep == 0) {
            if (mPIInfo.rank == 0) {
                std::cout << std::to_string(step) << ","
                          << std::setprecision(6) << totalTime
                          << std::endl;
                logfile << std::to_string(step) << ","
                        << std::setprecision(6) << totalTime
                        << std::endl;
            }
            idealMHD1D.save(directoryname, filenameWithoutStep, step);
        }

        idealMHD1D.oneStepRK2();
        
        if (mPIInfo.rank == 0) {
            totalTime += dt;
        }
    }
    
    MPI_Finalize();

    if (mPIInfo.rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    return 0;
}


