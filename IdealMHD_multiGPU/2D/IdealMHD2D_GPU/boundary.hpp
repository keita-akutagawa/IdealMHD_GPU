#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "mpi.hpp"


class Boundary
{
private:
    MPIInfo mPIInfo; 

public:
    Boundary(MPIInfo& mPIInfo);

    void periodicBoundaryX2nd(
        thrust::device_vector<ConservationParameter>& U
    );

    void periodicBoundaryY2nd(
        thrust::device_vector<ConservationParameter>& U
    );

    /*
    void symmetricBoundaryX2nd(
        thrust::device_vector<ConservationParameter>& U, 
        MPIInfo& mPIInfo
    );

    void symmetricBoundaryY2nd(
        thrust::device_vector<ConservationParameter>& U, 
        MPIInfo& mPIInfo
    );
    */

private:

};


