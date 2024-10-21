#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "flux.hpp"
#include "mpi.hpp"


class Boundary
{
private:
    MPIInfo mPIInfo; 
    MPIInfo* device_mPIInfo; 

public:
    Boundary(MPIInfo& mPIInfo);

    void periodicBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryX2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );

    void periodicBoundaryY2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );

    /*
    void symmetricBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void symmetricBoundaryX2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );
    */

    void symmetricBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void symmetricBoundaryY2nd_flux(
        thrust::device_vector<Flux>& fluxF, 
        thrust::device_vector<Flux>& fluxG
    );

private:

};


