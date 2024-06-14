#include <thrust/device_vector.h>
#include "const.hpp"
#include "minmod.hpp"


class MUSCL
{
public:
    void getLeftComponent(
        const thrust::device_vector<double> q, 
        thrust::device_vector<double>& qLeft
    );
    void getRightComponent(
        const thrust::device_vector<double> q, 
        thrust::device_vector<double>& qRight
    );
};

