#include "const.hpp"
#include "hlld.hpp"


class FluxSolver
{
private:
    HLLD hLLD;
    thrust::device_vector<Flux> flux;

public:
    thrust::device_vector<Flux> getFluxF(
        const thrust::device_vector<ConservationParameter>& U
    );

    thrust::device_vector<Flux> getFluxG(
        const thrust::device_vector<ConservationParameter>& U
    );
};


