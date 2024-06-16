#include <thrust/device_vector.h>
#include "const.hpp"
#include "muscl.hpp"
#include "conservation_parameter_struct.hpp"
#include "basic_parameter_struct.hpp"



class CalculateHalfComponents
{
private:
    thrust::device_vector<BasicParameter> qCenter;
    thrust::device_vector<BasicParameter> qLeft;
    thrust::device_vector<BasicParameter> qRight;

    MUSCL muscl;

public:

    void setPhysicalParameters(
        const thrust::device_vector<ConservationParameter>& U
    );

    void calculateLeftComponents();

    void calculateRightComponents();

    thrust::device_vector<BasicParameter> getCenterComponents();

    thrust::device_vector<BasicParameter> getLeftComponents();

    thrust::device_vector<BasicParameter> getRightComponents();
};

