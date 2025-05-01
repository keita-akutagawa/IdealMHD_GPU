#include "main_steady_const.hpp"

__global__ void initializeU_kernel(
    ConservationParameter* U, 
    double rho0, double v0, double B0, double p0, 
    MPIInfo* device_mPIInfo
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < device_nx && j < device_ny) {

        if (device_mPIInfo->isInside(i, j)) {
            int index = device_mPIInfo->globalToLocal(i, j);

            double rho, u, v, w, bX, bY, bZ, e, p;
            int buffer = device_mPIInfo->buffer;  
            double x = (i - buffer) * device_dx + device_xmin; 
            
            rho = rho0;
            u   = 0.0;
            v   = 0.0;
            w   = 0.0;
            bX  = 0.0;
            bY  = B0 / x;
            bZ  = 0.0;
            p   = p0;
            e   = p / (device_gamma_mhd - 1.0)
                + 0.5 * rho * (u * u + v * v + w * w)
                + 0.5 * (bX * bX + bY * bY + bZ * bZ);

            U[index].rho  = rho;
            U[index].rhoU = rho * u;
            U[index].rhoV = rho * v;
            U[index].rhoW = rho * w;
            U[index].bX   = bX;
            U[index].bY   = bY;
            U[index].bZ   = bZ;
            U[index].e    = e;
        }
    }
}

void IdealMHD2D::initializeU()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    double rho0 = 1.0; 
    double v0 = 0.0; 
    double B0 = 1.0; 
    double p0 = 1.0; 
    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        rho0, v0, B0, p0, 
        device_mPIInfo
    );
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);

    boundary.symmetricBoundaryX2nd_U(U);
    boundary.periodicBoundaryY2nd_U(U);

    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    
    MPIInfo mPIInfo;
    setupInfo(mPIInfo, buffer);

    if (mPIInfo.rank == 0) {
        std::cout   << mPIInfo.gridX << std::endl;
        mpifile << mPIInfo.gridX << std::endl;
    }

    initializeDeviceConstants();
    
    IdealMHD2D idealMHD2D(mPIInfo);

    idealMHD2D.initializeU(); 

    for (int step = 0; step < totalStep + 1; step++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (mPIInfo.rank == 0) {
            if (step % recordStep == 0) {
                std::cout << std::to_string(step) << " step done : total time is "
                        << std::setprecision(4) << totalTime << std::endl;
            }
        }

        if (step % recordStep == 0) {
            logfile << std::setprecision(6) << totalTime << std::endl;
            idealMHD2D.save(
                directoryName, filenameWithoutStep + "_U", step
            );
        }

        idealMHD2D.oneStepRK2_symmetricXperiodicY();

        if (idealMHD2D.checkCalculationIsCrashed()) {
            logfile << std::setprecision(6) << totalTime << std::endl;
            idealMHD2D.save(
                directoryName, filenameWithoutStep + "_U", step
            );
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
            break;
        }

        if (mPIInfo.rank == 0) {
            totalTime += dt;
        }   
    }

    if (mPIInfo.rank == 0) {
        std::cout << "program was completed!" << std::endl;
    }

    MPI_Finalize();

    return 0;
}



