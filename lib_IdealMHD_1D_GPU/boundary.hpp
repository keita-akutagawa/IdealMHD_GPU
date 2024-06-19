#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"


class Boundary
{
private:

public:

    void periodicBoundary(
        thrust::device_vector<ConservationParameter>& U
    );

    void symmetricBoundary2nd(
        thrust::device_vector<ConservationParameter>& U
    );

private:

};


