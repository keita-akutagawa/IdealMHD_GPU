#include <thrust/device_vector.h>
#include "const.hpp"
#include "muscl.hpp"
#include "conservation_parameter_struct.hpp"
#include "basic_parameter_struct.hpp"



class CalculateHalfQ
{
private:
    thrust::device_vector<BasicParameter> dQCenter;
    thrust::device_vector<BasicParameter> dQLeft;
    thrust::device_vector<BasicParameter> dQRight;

    MUSCL muscl;

public:

    void setPhysicalParameters(
        const thrust::device_vector<ConservationParameter>& U
    );

    void calculateLeftQ();

    void calculateRightQ();

    thrust::device_vector<BasicParameter> getCenterQ();

    thrust::device_vector<BasicParameter> getLeftQ();

    thrust::device_vector<BasicParameter> getRightQ();
};

