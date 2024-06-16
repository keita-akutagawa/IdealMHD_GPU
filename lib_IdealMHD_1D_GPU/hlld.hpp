#include <thrust/device_vector.h>
#include "const.hpp"
#include "calculate_half_Q.hpp"
#include "fan_parameter_struct.hpp"
#include "hlld_parameter_struct.hpp"
#include "flux_struct.hpp"


class HLLD
{
private:
    CalculateHalfQ calculateHalfQ;

    thrust::device_vector<BasicParameter> dQCenter;
    thrust::device_vector<BasicParameter> dQLeft;
    thrust::device_vector<BasicParameter> dQRight;
    thrust::device_vector<FanParameter> outerLeftFanParameter;
    thrust::device_vector<FanParameter> outerRightFanParameter;
    thrust::device_vector<FanParameter> middleLeftFanParameter;
    thrust::device_vector<FanParameter> middleRightFanParameter;
    thrust::device_vector<FanParameter> innerLeftFanParameter;
    thrust::device_vector<FanParameter> innerRightFanParameter;
    thrust::device_vector<HLLDParameter> hLLDLeftParameter;
    thrust::device_vector<HLLDParameter> hLLDRightParameter;

    thrust::device_vector<Flux> flux;
    thrust::device_vector<Flux> fluxOuterLeft, fluxMiddleLeft, fluxInnerLeft;
    thrust::device_vector<Flux> fluxOuterRight, fluxMiddleRight, fluxInnerRight;

public:

    void calculateFlux(
        const thrust::device_vector<ConservationParameter>& U
    );

    thrust::device_vector<Flux> getFlux();

private:
    double sign(double x);

    void setComponents(
        const thrust::device_vector<ConservationParameter>& U
    );

    void calculateHLLDParametersForOuterFan();

    void calculateHLLDParametersForMiddleFan();

    void calculateHLLDParametersForInnerFan();

    void setFanParametersFromComponents(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<FanParameter>& fanParameter
    );

    void calculateHLLDSubParametersForMiddleFan(
        const thrust::device_vector<BasicParameter>& components, 
        const thrust::device_vector<FanParameter>& outerFanParameter, 
        thrust::device_vector<HLLDParameter>& hLLDParameter
    );

    void calculateHLLDParameters1(
        const thrust::device_vector<FanParameter>& outerFanParameter, 
        const thrust::device_vector<HLLDParameter>& hLLDParameter, 
        thrust::device_vector<FanParameter>& middleFanParameter
    );

    void calculateHLLDParameters2();

    void setFlux(
        const thrust::device_vector<FanParameter>& fanParameter, 
        thrust::device_vector<Flux>& flux
    );
};

