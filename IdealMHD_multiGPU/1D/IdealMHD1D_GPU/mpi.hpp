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
    int gridX;
    int localGridX;
    int localNx; 
    int localSize; 

    int buffer = 1;

    MPI_Datatype mpi_conservation_parameter_type;
    MPI_Datatype mpi_flux_type;    


    __host__ __device__
    int getRank(int dx);

    __host__ __device__
    bool isInside(int globalX);

    __host__ __device__
    int globalToLocal(int globalX);
};


void setupInfo(MPIInfo& mPIInfo);


void sendrecv_U(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo);


void sendrecv_flux(thrust::device_vector<Flux>& flux, MPIInfo& mPIInfo);

#endif


