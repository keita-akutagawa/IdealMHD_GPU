#include "sign.hpp"

__host__ __device__
int sign(const double& x)
{
    return (x > 0.0) - (x < 0.0);
}

