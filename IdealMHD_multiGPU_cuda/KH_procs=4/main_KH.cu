#include "main_KH_const.hpp"


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

            double xPosition, yPosition;
            xPosition = i * device_dx - device_xCenter;
            yPosition = j * device_dy - device_yCenter;

            double rho, u, v, w, bx, by, bz, p, e;
            rho = device_rho0 / 2.0 * ((1.0 - device_rr) * tanh(yPosition / device_shear_thickness) + 1.0 + device_rr);
            u = -device_v0 / 2.0 * tanh(yPosition / device_shear_thickness);
            v = 0.02 * device_v0 * cos(2.0 * device_PI * xPosition / device_xmax) / pow(cosh(yPosition / device_shear_thickness), 2);
            w = 0.0;
            bx = device_b0 / 2.0 * ((1.0 - device_br) * tanh(yPosition / device_shear_thickness) + 1.0 + device_br) * cos(device_theta);
            by = 0.0;
            bz = device_b0 / 2.0 * ((1.0 - device_br) * tanh(yPosition / device_shear_thickness) + 1.0 + device_br) * sin(device_theta);
            p = device_beta * (bx * bx + by * by + bz * bz) / 2.0;
            e = p / (device_gamma_mhd - 1.0)
                + 0.5 * rho * (u * u + v * v + w * w)
                + 0.5 * (bx * bx + by * by + bz * bz);
            
            U[index].rho  = rho;
            U[index].rhoU = rho * u;
            U[index].rhoV = rho * v;
            U[index].rhoW = rho * w;
            U[index].bX   = bx;
            U[index].bY   = by;
            U[index].bZ   = bz;
            U[index].e    = e;
        }
    }
}

void IdealMHD2D::initializeU()
{
    cudaMemcpyToSymbol(device_xCenter, &xCenter, sizeof(double));
    cudaMemcpyToSymbol(device_yCenter, &yCenter, sizeof(double));
    cudaMemcpyToSymbol(device_shear_thickness, &shear_thickness, sizeof(double));
    cudaMemcpyToSymbol(device_rr, &rr, sizeof(double));
    cudaMemcpyToSymbol(device_br, &br, sizeof(double));
    cudaMemcpyToSymbol(device_beta, &beta, sizeof(double));
    cudaMemcpyToSymbol(device_theta, &theta, sizeof(double));
    cudaMemcpyToSymbol(device_rho0, &rho0, sizeof(double));
    cudaMemcpyToSymbol(device_b0, &b0, sizeof(double));
    cudaMemcpyToSymbol(device_p0, &p0, sizeof(double));
    cudaMemcpyToSymbol(device_v0, &v0, sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    initializeU_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        device_mPIInfo
    );
    cudaDeviceSynchronize();
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

    for (int step = 0; step < totalStep+1; step++) {
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
        
        idealMHD2D.oneStepRK2_periodicXWallY();

        if (idealMHD2D.checkCalculationIsCrashed()) {
            std::cout << "Calculation stopped! : " << step << " steps" << std::endl;
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


