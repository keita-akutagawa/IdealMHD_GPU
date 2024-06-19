#include "boundary.hpp"
#include "const.hpp"


struct PeriodicBoundaryFunctor {
    ConservationParameter conservationParameterLeft;
    ConservationParameter conservationParameterRight;

     PeriodicBoundaryFunctor(
        ConservationParameter Left, 
        ConservationParameter Right
    ) : conservationParameterLeft(Left), 
        conservationParameterRight(Right) 
        {}

    __device__
    void operator()(const ConservationParameter& conservationParameter) const {

    }
};

void Boundary::periodicBoundary(
    thrust::device_vector<ConservationParameter>& U
)
{
    ConservationParameter conservationParameterLeft;
    ConservationParameter conservationParameterRight;
    conservationParameterLeft = U.front();
    conservationParameterRight = U.back();

    thrust::transform(
        U.begin(), 
        U.end(), 
        U.begin(),
        PeriodicBoundaryFunctor(
            conservationParameterLeft, 
            conservationParameterRight
        )
    );
}


void Boundary::symmetricBoundary2nd(
    thrust::device_vector<ConservationParameter>& U
)
{
    for (int comp = 0; comp < 8; comp++) {
        U[comp][0] = U[comp][2];
        U[comp][1] = U[comp][2];
        U[comp][nx-1] = U[comp][nx-3];
        U[comp][nx-2] = U[comp][nx-3];
    }
}

