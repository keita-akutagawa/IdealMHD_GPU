#include "const.hpp"
#include "hlld.hpp"
#include "basic_parameter_struct.hpp"
#include "mpi.hpp"


class FluxSolver
{
private:
    MPIInfo mPIInfo;

    HLLD hLLD;
    thrust::device_vector<Flux> flux;
    thrust::device_vector<BasicParameter> dQLeft; 
    thrust::device_vector<BasicParameter> dQRight; 

public:
    FluxSolver(MPIInfo& mPIInfo);

    thrust::device_vector<Flux> getFluxF(
        const thrust::device_vector<ConservationParameter>& U
    );

    thrust::device_vector<Flux> getFluxG(
        const thrust::device_vector<ConservationParameter>& U
    );
};


