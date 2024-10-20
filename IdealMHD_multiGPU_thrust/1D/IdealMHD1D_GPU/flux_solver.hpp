#include "const.hpp"
#include "hlld.hpp"


class FluxSolver
{
private:
    HLLD hLLD;
    thrust::device_vector<Flux> flux;

public:
    FluxSolver(int localSize);

    thrust::device_vector<Flux>& getFluxF(
        const thrust::device_vector<ConservationParameter>& U
    );
};


