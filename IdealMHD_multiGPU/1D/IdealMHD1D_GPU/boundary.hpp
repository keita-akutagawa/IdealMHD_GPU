#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "mpi.hpp"


class Boundary
{
private:

public:

    void symmetricBoundary2nd(
        thrust::device_vector<ConservationParameter>& U, 
        MPIInfo& mPIInfo
    );

private:

};


