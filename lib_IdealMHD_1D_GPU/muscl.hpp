#include <thrust/device_vector.h>
#include "const.hpp"
#include "minmod.hpp"


class MUSCL
{
public:

    void getLeftComponent(
        const thrust::device_vector<double>& dQ, 
        thrust::device_vector<double>& dQLeft
    );
    void getRightComponent(
        const thrust::device_vector<double>& dQ, 
        thrust::device_vector<double>& dQRight
    );
};

