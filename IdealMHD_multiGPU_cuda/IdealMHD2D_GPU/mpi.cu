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
    mPIInfo.localGridX = rank / mPIInfo.gridX;
    mPIInfo.localGridY = rank % mPIInfo.gridX;
    mPIInfo.localNx = nx / mPIInfo.gridX;
    mPIInfo.localNy = ny / mPIInfo.gridY;
    mPIInfo.buffer = buffer;
    mPIInfo.localSizeX = mPIInfo.localNx + 2 * mPIInfo.buffer;
    mPIInfo.localSizeY = mPIInfo.localNy + 2 * mPIInfo.buffer;


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


void sendrecv_U_x(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    //int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int left = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    thrust::host_vector<ConservationParameter> sendULeft(mPIInfo.buffer * localNy), sendURight(mPIInfo.buffer * localNy);
    thrust::host_vector<ConservationParameter> recvULeft(mPIInfo.buffer * localNy), recvURight(mPIInfo.buffer * localNy);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            sendURight[j + i * localNy] = U[j + mPIInfo.buffer + (localNx + i)        * localSizeY];
            sendULeft[ j + i * localNy] = U[j + mPIInfo.buffer + (mPIInfo.buffer + i) * localSizeY];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Sendrecv(sendURight.data(), sendURight.size(), mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 recvULeft.data(),  recvULeft.size(),  mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendULeft.data(),  sendULeft.size(),  mPIInfo.mpi_conservation_parameter_type, left,  0, 
                 recvURight.data(), recvURight.size(), mPIInfo.mpi_conservation_parameter_type, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            U[j + mPIInfo.buffer + i                              * localSizeY] = recvULeft[ j + i * localNy];
            U[j + mPIInfo.buffer + (localNx + mPIInfo.buffer + i) * localSizeY] = recvURight[j + i * localNy];
        }
    }
}


void sendrecv_U_y(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo)
{
    //int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int up = mPIInfo.getRank(0, -1);
    int down = mPIInfo.getRank(0, 1);
    MPI_Status st;

    thrust::host_vector<ConservationParameter> sendUUp(mPIInfo.buffer * localSizeX), sendUDown(mPIInfo.buffer * localSizeX);
    thrust::host_vector<ConservationParameter> recvUUp(mPIInfo.buffer * localSizeX), recvUDown(mPIInfo.buffer * localSizeX);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            sendUDown[j + i * mPIInfo.buffer] = U[j + localNy        + i * localSizeY];
            sendUUp[  j + i * mPIInfo.buffer] = U[j + mPIInfo.buffer + i * localSizeY];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Sendrecv(sendUDown.data(), sendUDown.size(), mPIInfo.mpi_conservation_parameter_type, down, 0, 
                 recvUUp.data(),   recvUUp.size(),   mPIInfo.mpi_conservation_parameter_type, up,   0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendUUp.data(),   sendUUp.size(),   mPIInfo.mpi_conservation_parameter_type, up,   0, 
                 recvUDown.data(), recvUDown.size(), mPIInfo.mpi_conservation_parameter_type, down, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            U[j                            + i * localSizeY] = recvUUp[  j + i * mPIInfo.buffer];
            U[j + localNy + mPIInfo.buffer + i * localSizeY] = recvUDown[j + i * mPIInfo.buffer];
        }
    }
}


void sendrecv_U(thrust::device_vector<ConservationParameter>& U, MPIInfo& mPIInfo)
{
    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_U_x(U, mPIInfo);
    sendrecv_U_y(U, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
}

//////////

void sendrecv_flux_x(thrust::device_vector<Flux>& flux, MPIInfo& mPIInfo)
{
    int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    //int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int left = mPIInfo.getRank(-1, 0);
    int right = mPIInfo.getRank(1, 0);
    MPI_Status st;

    thrust::host_vector<Flux> sendFluxLeft(mPIInfo.buffer * localNy), sendFluxRight(mPIInfo.buffer * localNy);
    thrust::host_vector<Flux> recvFluxLeft(mPIInfo.buffer * localNy), recvFluxRight(mPIInfo.buffer * localNy);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            sendFluxRight[j + i * localNy] = flux[j + mPIInfo.buffer + (localNx + i)        * localSizeY];
            sendFluxLeft[ j + i * localNy] = flux[j + mPIInfo.buffer + (mPIInfo.buffer + i) * localSizeY];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Sendrecv(sendFluxRight.data(), sendFluxRight.size(), mPIInfo.mpi_flux_type, right, 0, 
                 recvFluxLeft.data(),  recvFluxLeft.size(),  mPIInfo.mpi_flux_type, left,  0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendFluxLeft.data(),  sendFluxLeft.size(),  mPIInfo.mpi_flux_type, left,  0, 
                 recvFluxRight.data(), recvFluxRight.size(), mPIInfo.mpi_flux_type, right, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < mPIInfo.buffer; i++) {
        for (int j = 0; j < localNy; j++) {
            flux[j + mPIInfo.buffer + i                              * localSizeY] = recvFluxLeft[ j + i * localNy];
            flux[j + mPIInfo.buffer + (localNx + mPIInfo.buffer + i) * localSizeY] = recvFluxRight[j + i * localNy];
        }
    }
}


void sendrecv_flux_y(thrust::device_vector<Flux>& flux, MPIInfo& mPIInfo)
{
    //int localNx = mPIInfo.localNx;
    int localNy = mPIInfo.localNy;
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    int up = mPIInfo.getRank(0, -1);
    int down = mPIInfo.getRank(0, 1);
    MPI_Status st;

    thrust::host_vector<Flux> sendFluxUp(mPIInfo.buffer * localSizeX), sendFluxDown(mPIInfo.buffer * localSizeX);
    thrust::host_vector<Flux> recvFluxUp(mPIInfo.buffer * localSizeX), recvFluxDown(mPIInfo.buffer * localSizeX);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            sendFluxDown[j + i * mPIInfo.buffer] = flux[j + localNy        + i * localSizeY];
            sendFluxUp[  j + i * mPIInfo.buffer] = flux[j + mPIInfo.buffer + i * localSizeY];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Sendrecv(sendFluxDown.data(), sendFluxDown.size(), mPIInfo.mpi_flux_type, down, 0, 
                 recvFluxUp.data(),   recvFluxUp.size(),   mPIInfo.mpi_flux_type, up,   0, 
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(sendFluxUp.data(),   sendFluxUp.size(),   mPIInfo.mpi_flux_type, up,   0, 
                 recvFluxDown.data(), recvFluxDown.size(), mPIInfo.mpi_flux_type, down, 0, 
                 MPI_COMM_WORLD, &st);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < localSizeX; i++) {
        for (int j = 0; j < mPIInfo.buffer; j++) {
            flux[j                            + i * localSizeY] = recvFluxUp[  j + i * mPIInfo.buffer];
            flux[j + localNy + mPIInfo.buffer + i * localSizeY] = recvFluxDown[j + i * mPIInfo.buffer];
        }
    }
}


void sendrecv_flux(thrust::device_vector<Flux>& flux, MPIInfo& mPIInfo)
{
    MPI_Barrier(MPI_COMM_WORLD);
    sendrecv_flux_x(flux, mPIInfo);
    sendrecv_flux_y(flux, mPIInfo);
    MPI_Barrier(MPI_COMM_WORLD);
}

