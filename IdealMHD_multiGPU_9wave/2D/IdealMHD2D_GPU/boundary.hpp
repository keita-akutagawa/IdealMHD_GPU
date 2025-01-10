#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "flux_struct.hpp"
#include "mpi.hpp"


class Boundary
{
private:
    MPIInfo mPIInfo; 
    MPIInfo* device_mPIInfo; 

    thrust::device_vector<ConservationParameter> sendULeft; 
    thrust::device_vector<ConservationParameter> sendURight; 
    thrust::device_vector<ConservationParameter> recvULeft; 
    thrust::device_vector<ConservationParameter> recvURight; 

    thrust::device_vector<ConservationParameter> sendUDown; 
    thrust::device_vector<ConservationParameter> sendUUp; 
    thrust::device_vector<ConservationParameter> recvUDown; 
    thrust::device_vector<ConservationParameter> recvUUp; 


public:
    Boundary(MPIInfo& mPIInfo);

    void periodicBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    /*
    void wallBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );
    */

    void wallBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

    /*
    void symmetricBoundaryX2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );
    */

    void symmetricBoundaryY2nd_U(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


