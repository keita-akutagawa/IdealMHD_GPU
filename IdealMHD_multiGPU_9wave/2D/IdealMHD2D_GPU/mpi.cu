#include "mpi.hpp"


int MPIInfo::getRank(int dx, int dy)
{
    int rankX = (localGridX + dx + gridX) % gridX;
    int rankY = (localGridY + dy + gridY) % gridY;
    return rankY + rankX * gridY;
}


bool MPIInfo::isInside(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int endX = startX + localNx;
    int startY = localNy * localGridY;
    int endY = startY + localNy;

    if (globalX < startX) return false;
    if (globalX >= endX) return false;
    if (globalY < startY) return false;
    if (globalY >= endY) return false;

    return true;
}


int MPIInfo::globalToLocal(int globalX, int globalY)
{
    int startX = localNx * localGridX;
    int x = globalX - startX;

    int startY = localNy * localGridY;
    int y = globalY - startY;

    return y + buffer + (x + buffer) * localSizeY;
}


void setupInfo(MPIInfo& mPIInfo, int buffer)
{
    int rank = 0, procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    int d2[2] = {};
    MPI_Dims_create(procs, 2, d2);
    mPIInfo.rank = rank;
    mPIInfo.procs = procs;
    mPIInfo.gridX = d2[0];
    mPIInfo.gridY = d2[1];
    mPIInfo.localGridX = rank / mPIInfo.gridY;
    mPIInfo.localGridY = rank % mPIInfo.gridY;
    mPIInfo.localNx = nx / mPIInfo.gridX;
    mPIInfo.localNy = ny / mPIInfo.gridY;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
    mPIInfo.localSizeY = mPIInfo.localNy + 2 * mPIInfo.buffer;


    int block_lengths_conservation_parameter[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint offsets_conservation_parameter[9];
    offsets_conservation_parameter[0] = offsetof(ConservationParameter, rho);
    offsets_conservation_parameter[1] = offsetof(ConservationParameter, rhoU);
    offsets_conservation_parameter[2] = offsetof(ConservationParameter, rhoV);
    offsets_conservation_parameter[3] = offsetof(ConservationParameter, rhoW);
    offsets_conservation_parameter[4] = offsetof(ConservationParameter, bX);
    offsets_conservation_parameter[5] = offsetof(ConservationParameter, bY);
    offsets_conservation_parameter[6] = offsetof(ConservationParameter, bZ);
    offsets_conservation_parameter[7] = offsetof(ConservationParameter, e);
    offsets_conservation_parameter[8] = offsetof(ConservationParameter, psi);

    MPI_Datatype types_conservation_parameter[9] = {
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE
    };

    MPI_Type_create_struct(9, block_lengths_conservation_parameter, offsets_conservation_parameter, types_conservation_parameter, &mPIInfo.mpi_conservation_parameter_type);
    MPI_Type_commit(&mPIInfo.mpi_conservation_parameter_type);

}


void sendrecv_U_x(
    thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<ConservationParameter>& sendULeft, 
    thrust::device_vector<ConservationParameter>& sendURight, 
    thrust::device_vector<ConservationParameter>& recvULeft, 
    thrust::device_vector<ConservationParameter>& recvURight, 
    MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    //int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int left = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            sendULeft[ j + i * localNy] = U[j + mPIInfo.buffer + (mPIInfo.buffer + i) * localSizeY];
            sendURight[j + i * localNy] = U[j + mPIInfo.buffer + (localNx + i)        * localSizeY];
        }
    }

    MPI_Sendrecv(thrust::raw_pointer_cast(sendULeft.data()),  sendULeft.size(),  mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 thrust::raw_pointer_cast(recvURight.data()), recvURight.size(), mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(thrust::raw_pointer_cast(sendURight.data()), sendURight.size(), mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 thrust::raw_pointer_cast(recvULeft.data()),  recvULeft.size(),  mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 MPI_COMM_WORLD, &st);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            U[j + mPIInfo.buffer + i                              * localSizeY] = recvULeft[ j + i * localNy];
            U[j + mPIInfo.buffer + (localNx + mPIInfo.buffer + i) * localSizeY] = recvURight[j + i * localNy];
        }
    }
}


void sendrecv_U_y(
    thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<ConservationParameter>& sendUDown, 
    thrust::device_vector<ConservationParameter>& sendUUp, 
    thrust::device_vector<ConservationParameter>& recvUDown, 
    thrust::device_vector<ConservationParameter>& recvUUp, 
    MPIInfo& mPIInfo)
{
    //int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int down = mPIInfo.getRank(0, -1);
    int up   = mPIInfo.getRank(0, 1);
    MPI_Status st;

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            sendUDown[j + i * mPIInfo.buffer] = U[j + mPIInfo.buffer + i * localSizeY];
            sendUUp[  j + i * mPIInfo.buffer] = U[j + localNy        + i * localSizeY];
        }
    }

    MPI_Sendrecv(thrust::raw_pointer_cast(sendUDown.data()), sendUDown.size(), mPIInfo.mpi_conservation_parameter_type, down, 0, 
                 thrust::raw_pointer_cast(recvUUp.data()),   recvUUp.size(),   mPIInfo.mpi_conservation_parameter_type, up,   0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(thrust::raw_pointer_cast(sendUUp.data()),   sendUUp.size(),   mPIInfo.mpi_conservation_parameter_type, up,   0, 
                 thrust::raw_pointer_cast(recvUDown.data()), recvUDown.size(), mPIInfo.mpi_conservation_parameter_type, down, 0, 
                 MPI_COMM_WORLD, &st);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            U[j                            + i * localSizeY] = recvUDown[j + i * mPIInfo.buffer];
            U[j + localNy + mPIInfo.buffer + i * localSizeY] = recvUUp[  j + i * mPIInfo.buffer];
        }
    }
}



