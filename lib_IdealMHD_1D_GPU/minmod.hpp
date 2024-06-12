#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cmath>
#include "const.hpp"


struct MinMod
{
    __host__ __device__
    double operator()(const double& x, const double& y) const
    {
        int sign_x = (x > 0) - (x < 0);
        double abs_x = std::abs(x);

        return sign_x * thrust::max(thrust::min(abs_x, sign_x * y), device_EPS);
    }
};


