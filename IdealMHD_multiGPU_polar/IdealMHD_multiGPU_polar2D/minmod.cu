#include "minmod.hpp"


__host__ __device__
double minmod(const double& x, const double& y)
{
    int sign_x = (x > 0) - (x < 0);
    double abs_x = std::abs(x);

    return sign_x * thrust::max(thrust::min(abs_x, sign_x * y), device_EPS);
}

