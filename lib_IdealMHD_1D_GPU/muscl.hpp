#include <thrust/device_vector.h>
#include "const.hpp"
#include "minmod.hpp"


class MUSCL
{
private:
    thrust::device_vector<double> tmpQ1;
    thrust::device_vector<double> tmpQ2;

public:
    MUSCL() : tmpQ1(nx), tmpQ2(nx) {}

    void getLeftComponent(
        const thrust::device_vector<double>& dQ, 
        thrust::device_vector<double>& dQLeft
    );
    void getRightComponent(
        const thrust::device_vector<double>& dQ, 
        thrust::device_vector<double>& dQRight
    );
};

