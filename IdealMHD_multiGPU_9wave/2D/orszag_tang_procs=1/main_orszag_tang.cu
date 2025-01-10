#include "main_orszag_tang_const.hpp"


__global__ void initializeU_kernel(
    ConservationParameter* U, 
    MPIInfo* device_mPIInfo
) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {
        MPIInfo mPIInfo = *device_mPIInfo;

        if (mPIInfo.isInside(i, j)) {
            int index = mPIInfo.globalToLocal(i, j);

            double rho0 = device_gamma_mhd * device_gamma_mhd;
            double u0 = -sin(j * device_dy);
            double v0 = sin(i * device_dx);
            double w0 = 0.0;
            double bx0 = -sin(j * device_dy);
            double by0 = sin(2.0 * i * device_dx);
            double bz0 = 0.0;
            double p0 = device_gamma_mhd;
            double e0 = p0 / (device_gamma_mhd - 1.0)
                    + 0.5 * rho0 * (u0 * u0 + v0 * v0 + w0 * w0)
                    + 0.5 * (bx0 * bx0 + by0 * by0 + bz0 * bz0);
            
            U[index].rho  = rho0;
            U[index].rhoU = rho0 * u0;
            U[index].rhoV = rho0 * v0;
            U[index].rhoW = rho0 * w0;
            U[index].bX   = bx0;
            U[index].bY   = by0;
            U[index].bZ   = bz0;
            U[index].e    = e0;
        }
    }
}

void IdealMHD2D::initializeU()
{

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    boundary.periodicBoundaryX2nd_U(U);
    boundary.periodicBoundaryY2nd_U(U);
    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPIInfo mPIInfo;
    setupInfo(mPIInfo, buffer);

    if (mPIInfo.rank == 0) {
        std::cout << mPIInfo.gridX << "," << mPIInfo.gridY << std::endl;
        logfile   << mPIInfo.gridX << "," << mPIInfo.gridY << std::endl;
    }

    cudaSetDevice(mPIInfo.rank);

    initializeDeviceConstants();


    IdealMHD2D idealMHD2D(mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);

    idealMHD2D.initializeU();

    if (mPIInfo.rank == 0) {
        size_t free_mem = 0;
        size_t total_mem = 0;
        cudaError_t status = cudaMemGetInfo(&free_mem, &total_mem);

        std::cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << std::endl;
    }

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
            idealMHD2D.save(directoryname, filenameWithoutStep, step);
        }

        idealMHD2D.oneStepRK2();

        if (idealMHD2D.checkCalculationIsCrashed()) {
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            idealMHD2D.save(directoryname, filenameWithoutStep, step);
            return 0;
        }
        
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



