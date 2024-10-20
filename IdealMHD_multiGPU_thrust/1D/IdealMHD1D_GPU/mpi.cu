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

    return x + buffer;
}


void setupInfo(MPIInfo& mPIInfo, int buffer)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = procs;
    mPIInfo.localGridX = rank;
    mPIInfo.localNx = nx / mPIInfo.gridX;
    mPIInfo.buffer = buffer;
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
}


void sendrecv_U(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int left = mPIInfo.getRank(-1);
    int right = mPIInfo.getRank(1);
    MPI_Status st;

    thrust::host_vector<ConservationParameter> sendULeft(mPIInfo.buffer), sendURight(mPIInfo.buffer);
    thrust::host_vector<ConservationParameter> recvULeft(mPIInfo.buffer), recvURight(mPIInfo.buffer);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        sendURight[i] = U[localNx + i];
        sendULeft[i]  = U[mPIInfo.buffer + i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Sendrecv(sendURight.data(), mPIInfo.buffer, mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 recvULeft.data(),  mPIInfo.buffer, mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 MPI_COMM_WORLD, &st
    );
    MPI_Sendrecv(sendULeft.data(),  mPIInfo.buffer, mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 recvURight.data(), mPIInfo.buffer, mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 MPI_COMM_WORLD, &st
    );
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        U[i]                            = recvULeft[i];
        U[localNx + mPIInfo.buffer + i] = recvURight[i];
    }
}

