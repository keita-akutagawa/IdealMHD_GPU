#include <algorithm>
#include <vector>
#include <iostream>
#include "minmod.hpp"
#include "const.hpp"
#include "muscl.hpp"


void MUSCL::getLeftComponent(
    const std::vector<double>& q, 
    std::vector<double>& qLeft
)
{
    for (int i = 1; i < nDirection-1; i++) {
        qLeft[i] = q[i] + 0.5 * minmod(q[i] - q[i-1], q[i+1] - q[i]);
    }

    //周期境界条件
    qLeft[0] = q[0] + 0.5 * minmod(
        q[0] - q[nDirection-1], q[1] - q[0]
        );
    qLeft[nDirection-1] = q[nDirection-1] + 0.5 * minmod(
        q[nDirection-1] - q[nDirection-2], q[0] - q[nDirection-1]
        );
}


void MUSCL::getRightComponent(
    const std::vector<double>& q, 
    std::vector<double>& qRight
)
{
    for (int i = 0; i < nDirection-2; i++) {
        qRight[i] = q[i+1] - 0.5 * minmod(q[i+1] - q[i], q[i+2] - q[i+1]);
    }

    //周期境界条件
    qRight[nDirection-2] = q[nDirection-1] - 0.5 * minmod(
        q[nDirection-1] - q[nDirection-2], q[0] - q[nDirection-1]
        );
    qRight[nDirection-1] = q[0] - 0.5 * minmod(
        q[0] - q[nDirection-1], q[1] - q[0]
        );
}

