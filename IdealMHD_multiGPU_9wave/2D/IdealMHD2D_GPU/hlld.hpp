#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include "const.hpp"
#include "calculate_Q.hpp"
#include "hlld_parameter_struct.hpp"
#include "flux_struct.hpp"
#include "sign.hpp"
#include "mpi.hpp"


class HLLD
{
private:
    MPIInfo mPIInfo;

    CalculateQ calculateQ;

    thrust::device_vector<BasicParameter> dQCenter;
    thrust::device_vector<BasicParameter> dQLeft;
    thrust::device_vector<BasicParameter> dQRight;
    thrust::device_vector<HLLDParameter> hLLDParameter;

    thrust::device_vector<Flux> flux;
    thrust::device_vector<Flux> fluxOuterLeft, fluxMiddleLeft, fluxInnerLeft;
    thrust::device_vector<Flux> fluxOuterRight, fluxMiddleRight, fluxInnerRight;

    thrust::device_vector<ConservationParameter> tmpUForFluxG;

public:
    HLLD(MPIInfo& mPIInfo);

    void calculateFluxF(
        const thrust::device_vector<ConservationParameter>& U
    );

    void calculateFluxG(
        const thrust::device_vector<ConservationParameter>& U
    );

    thrust::device_vector<Flux>& getFlux();

private:

    void setQX(
        const thrust::device_vector<ConservationParameter>& U
    );

    void setQY(
        const thrust::device_vector<ConservationParameter>& U
    );

    void calculateHLLDParameter();

    void setFlux();

    void shuffleForTmpUForFluxG(
        const thrust::device_vector<ConservationParameter>& U
    );

    void shuffleFluxG();
};

