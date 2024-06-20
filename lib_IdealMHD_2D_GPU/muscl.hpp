#include <thrust/device_vector.h>
#include "const.hpp"
#include "minmod.hpp"
#include "basic_parameter_struct.hpp"


class MUSCL
{
public:

    void getLeftComponent(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQLeft
    );
    void getRightComponent(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQRight
    );
};

