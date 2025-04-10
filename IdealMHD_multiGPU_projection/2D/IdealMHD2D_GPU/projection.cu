#include "projection.hpp"


// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause]]]
//
//modified by Keita Akutagawa [2025.4.10]
//


Projection::Projection(
    MPIInfo& mPIInfo, 
    std::string MTXfilename, 
    std::string jsonFilenameForSolver
)
    : mPIInfo(mPIInfo), 
      
      //今のところは各プロセスが全領域を持っていて、ランク0だけが解くことにする
      divB(nx * ny), 
      sum_divB(nx * ny), 
      psi(nx * ny)
{
    AMGX_initialize();

    AMGX_config_create_from_file(&config, jsonFilenameForSolver.c_str());
    AMGX_resources_create_simple(&resource, config);
    AMGX_solver_create(&solver, resource, AMGX_mode_dDDI, config);

    AMGX_matrix_create(&A, resource, AMGX_mode_dDDI);
    AMGX_vector_create(&amgx_sol, resource, AMGX_mode_dDDI);
    AMGX_vector_create(&amgx_rhs, resource, AMGX_mode_dDDI);

    AMGX_read_system(A, amgx_sol, amgx_rhs, MTXfilename.c_str());

    AMGX_solver_setup(solver, A);
}


Projection::~Projection()
{
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(amgx_rhs);
    AMGX_vector_destroy(amgx_sol);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(resource);
    AMGX_config_destroy(config);
    AMGX_finalize();
}


__global__ void calculateDivB_kernel(
    double* divB, 
    const ConservationParameter* U, 
    int localNx, int localNy, int buffer, 
    int localSizeX, int localSizeY, 
    int localGridX, int localGridY
)   
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNx && j < localNy) {
        int index = j + buffer
                  + (i + buffer) * localSizeY;
        int indexForDivB = j + localNy * localGridY
                         + (i + localNx * localGridX) * device_ny; 
        
        divB[indexForDivB] = (U[index + localSizeY].bX - U[index - localSizeY].bX) / (2.0 * device_dx)
                           + (U[index + 1].bY - U[index - 1].bY) / (2.0 * device_dy);
        divB[indexForDivB] = -1.0 * divB[indexForDivB]; //正定値行列問題にするため、符号を反転させる。
    }
}


__global__ void correctDivB_kernel(
    ConservationParameter* U, 
    const double* psi, 
    int localNx, int localNy, int buffer, 
    int localSizeX, int localSizeY, 
    int localGridX, int localGridY
)   
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localNx && j < localNy) {
        int index = j + buffer
                  + (i + buffer) * localSizeY;
        int indexForPsi = j + localNy * localGridY
                        + (i + localNx * localGridX) * device_ny; 
        
        U[index].bX -= (psi[indexForPsi + device_ny] - psi[indexForPsi - device_ny]) / (2.0 * device_dx);
        U[index].bY -= (psi[indexForPsi + 1] - psi[indexForPsi - 1]) / (2.0 * device_dy);
    }
}



void Projection::correctB(
    thrust::device_vector<ConservationParameter>& U
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localNx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localNy + threadsPerBlock.y - 1) / threadsPerBlock.y);

    calculateDivB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(divB.data()), 
        thrust::raw_pointer_cast(U.data()),
        mPIInfo.localNx, mPIInfo.localNy, mPIInfo.buffer, 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.localGridX, mPIInfo.localGridY
    );
    cudaDeviceSynchronize();

    MPI_Reduce(thrust::raw_pointer_cast(divB.data()), thrust::raw_pointer_cast(sum_divB.data()), nx * ny, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mPIInfo.rank == 0) {
        AMGX_vector_upload(amgx_rhs, nx * ny, 1, thrust::raw_pointer_cast(sum_divB.data()));
        AMGX_solver_solve(solver, amgx_rhs, amgx_sol);
        AMGX_vector_download(amgx_sol, thrust::raw_pointer_cast(psi.data()));
    }
    MPI_Bcast(thrust::raw_pointer_cast(psi.data()), nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    correctDivB_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(psi.data()),
        mPIInfo.localNx, mPIInfo.localNy, mPIInfo.buffer, 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 
        mPIInfo.localGridX, mPIInfo.localGridY
    );
    cudaDeviceSynchronize();
}


