#include "boundary.hpp"
#include "const.hpp"
#include <iostream>


void Boundary::periodicBoundary(
    std::vector<std::vector<std::vector<double>>>& U
)
{
    //他のクラスの関数は周期境界を想定して組んでいるので
    //何もしなくてよい
    return;
}


void Boundary::symmetricBoundary2ndX(
    std::vector<std::vector<std::vector<double>>>& U
)
{
    for (int comp = 0; comp < 8; comp++) {
        for (int j = 0; j < ny; j++) {
            U[comp][0][j] = U[comp][3][j];
            U[comp][1][j] = U[comp][3][j];
            U[comp][2][j] = U[comp][3][j];
            U[comp][nx-1][j] = U[comp][nx-4][j];
            U[comp][nx-2][j] = U[comp][nx-4][j];
            U[comp][nx-3][j] = U[comp][nx-4][j];
        }
    }
}


void Boundary::symmetricBoundary2ndY(
    std::vector<std::vector<std::vector<double>>>& U
)
{
    for (int comp = 0; comp < 5; comp++) {
        for (int i = 0; i < nx; i++) {
            U[comp][i][0] = U[comp][i][5];
            U[comp][i][1] = U[comp][i][4];
            U[comp][i][2] = U[comp][i][3];
            U[comp][i][ny-1] = U[comp][i][ny-6];
            U[comp][i][ny-2] = U[comp][i][ny-5];
            U[comp][i][ny-3] = U[comp][i][ny-4];
        }
    }

    //Byは半整数格子点上にある
    for (int i = 0; i < nx; i++) {
        U[5][i][0] = U[5][i][4];
        U[5][i][1] = U[5][i][3];
        U[5][i][ny-1] = U[5][i][ny-5];
        U[5][i][ny-2] = U[5][i][ny-4];
    }

    for (int comp = 6; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            U[comp][i][0] = U[comp][i][5];
            U[comp][i][1] = U[comp][i][4];
            U[comp][i][2] = U[comp][i][3];
            U[comp][i][ny-1] = U[comp][i][ny-6];
            U[comp][i][ny-2] = U[comp][i][ny-5];
            U[comp][i][ny-3] = U[comp][i][ny-4];
        }
    }
}

