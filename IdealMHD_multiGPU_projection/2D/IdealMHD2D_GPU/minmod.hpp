#ifndef MINMOD_H
#define MINMOD_H

#include <thrust/device_vector.h>
#include <cmath>
#include "const.hpp"


__host__ __device__
double minmod(const double& x, const double& y);

#endif
