#ifndef MPI_H
#define MPI_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <mpi.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "flux_struct.hpp"


struct MPIInfo
{
    int rank;
    int procs;
    int gridX, gridY;
    int localGridX, localGridY;
    int localNx, localNy; 
    int buffer;
    int localSizeX, localSizeY; 

    MPI_Datatype mpi_conservation_parameter_type;
    MPI_Datatype mpi_flux_type;    


    __host__ __device__
    int getRank(int dx, int dy);

    __host__ __device__
    bool isInside(int globalX, int globalY);

    __host__ __device__
    int globalToLocal(int globalX, int globalY);
};


void setupInfo(MPIInfo& mPIInfo, int buffer);


void sendrecv_U_x(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo);

void sendrecv_U_y(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo);

void sendrecv_U(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo);


void sendrecv_flux_x(thrust::device_vector<Flux>& flux, MPIInfo& mPIInfo);

void sendrecv_flux_y(thrust::device_vector<Flux>& flux, MPIInfo& mPIInfo);

void sendrecv_flux(thrust::device_vector<Flux>& flux, MPIInfo& mPIInfo);


#endif
