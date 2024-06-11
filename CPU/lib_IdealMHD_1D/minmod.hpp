#include <algorithm>
#include "const.hpp"


inline double minmod(double x, double y)
{
    int sign_x = (x > 0) - (x < 0);
    double abs_x = std::abs(x);

    return sign_x * std::max(std::min(abs_x, sign_x * y), EPS);
}


