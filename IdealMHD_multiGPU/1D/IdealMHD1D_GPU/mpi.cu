#include "mpi.hpp"


int MPIInfo::getRank(int dx)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    return rankX;
}


bool MPIInfo::isInside(int globalX)
{
    int startX = localNx * localGridX;
    int endX = startX + localNx;

    if (globalX < startX) return false;
    if (globalX >= endX) return false;

    return true;
}


int MPIInfo::globalToLocal(int globalX)
{
    int startX = localNx * localGridX;

    int x = globalX - startX;

    return x + 1;
}


void setupInfo(MPIInfo& mPIInfo)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = procs;
    mPIInfo.localGridX = rank;
    mPIInfo.localNx = nx / mPIInfo.gridX;
    mPIInfo.localSize = mPIInfo.localNx + 2 * mPIInfo.buffer;


    int block_lengths_conservation_parameter[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_conservation_parameter[8];
    offsets_conservation_parameter[0] = offsetof(ConservationParameter, rho);
    offsets_conservation_parameter[1] = offsetof(ConservationParameter, rhoU);
    offsets_conservation_parameter[2] = offsetof(ConservationParameter, rhoV);
    offsets_conservation_parameter[3] = offsetof(ConservationParameter, rhoW);
    offsets_conservation_parameter[4] = offsetof(ConservationParameter, bX);
    offsets_conservation_parameter[5] = offsetof(ConservationParameter, bY);
    offsets_conservation_parameter[6] = offsetof(ConservationParameter, bZ);
    offsets_conservation_parameter[7] = offsetof(ConservationParameter, e);

    MPI_Datatype types_conservation_parameter[8] = {
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE
    };

    MPI_Type_create_struct(8, block_lengths_conservation_parameter, offsets_conservation_parameter, types_conservation_parameter, &mPIInfo.mpi_conservation_parameter_type);
    MPI_Type_commit(&mPIInfo.mpi_conservation_parameter_type);


    int block_lengths_flux[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_flux[8];
    offsets_flux[0] = offsetof(Flux, f0);
    offsets_flux[1] = offsetof(Flux, f1);
    offsets_flux[2] = offsetof(Flux, f2);
    offsets_flux[3] = offsetof(Flux, f3);
    offsets_flux[4] = offsetof(Flux, f4);
    offsets_flux[5] = offsetof(Flux, f5);
    offsets_flux[6] = offsetof(Flux, f6);
    offsets_flux[7] = offsetof(Flux, f7);

    MPI_Datatype types_flux[8] = {
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE
    };

    MPI_Type_create_struct(8, block_lengths_flux, offsets_flux, types_flux, &mPIInfo.mpi_flux_type);
    MPI_Type_commit(&mPIInfo.mpi_flux_type);
}


void sendrecv_U(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int left = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;

    ConservationParameter sendULeft, sendURight;
    ConservationParameter recvULeft, recvURight;

    sendURight = U[localNx];
    sendULeft  = U[1];

    MPI_Sendrecv(&(sendURight), 1, mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 &(recvULeft),  1, mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 MPI_COMM_WORLD, &st
    );
    MPI_Sendrecv(&(sendULeft),  1, mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 &(recvURight), 1, mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 MPI_COMM_WORLD, &st
    );

    U[0]           = recvULeft;
    U[localNx + 1] = recvURight;
}


void sendrecv_flux(thrust::device_vector<Flux>& flux, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int left = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;

    Flux sendFluxLeft, sendFluxRight;
    Flux recvFluxLeft, recvFluxRight;

    sendFluxRight = flux[localNx];
    sendFluxLeft  = flux[1];

    MPI_Sendrecv(&(sendFluxRight), 1, mPIInfo.mpi_flux_type, right, 0, 
                 &(recvFluxLeft),  1, mPIInfo.mpi_flux_type, left,  0, 
                 MPI_COMM_WORLD, &st
    );
    MPI_Sendrecv(&(sendFluxLeft),  1, mPIInfo.mpi_flux_type, left,  0, 
                 &(recvFluxRight), 1, mPIInfo.mpi_flux_type, right, 0, 
                 MPI_COMM_WORLD, &st
    );

    flux[0]           = recvFluxLeft;
    flux[localNx + 1] = recvFluxRight;
}


