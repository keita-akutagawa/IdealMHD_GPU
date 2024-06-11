#include "boundary.hpp"
#include "const.hpp"


void Boundary::periodicBoundary(
    std::vector<std::vector<double>>& U
)
{
    //他のクラスの関数は周期境界を想定して組んでいるので
    //何もしなくてよい
    return;
}


void Boundary::symmetricBoundary2nd(
    std::vector<std::vector<double>>& U
)
{
    for (int comp = 0; comp < 8; comp++) {
        U[comp][0] = U[comp][2];
        U[comp][1] = U[comp][2];
        U[comp][nx-1] = U[comp][nx-3];
        U[comp][nx-2] = U[comp][nx-3];
    }
}

