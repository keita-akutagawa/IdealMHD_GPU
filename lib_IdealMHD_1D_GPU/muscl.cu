#include "muscl.hpp"
#include <thrust/adjacent_difference.h>
#include <thrust/transform.h>


struct LeftComponentFunctor {
    MinMod minmod;

    __host__ __device__
    double operator()(const double q) const {
        return q;
    }
};

void MUSCL::getLeftComponent(
    const thrust::device_vector<double>& q, 
    thrust::device_vector<double>& qLeft
)
{
    thrust::counting_iterator<int> indices(0);
    MinMod minmod;

    thrust::adjacent_difference(
        q.begin(), 
        q.end(), 
        tmpQ.begin(), 
        thrust::minus<double>()
    );

    thrust::transform(
        q.begin() + 1, 
        q.end() - 1, 
        qLeft.begin() + 1, 
        LeftComponentFunctor()
    );

    // 周期境界条件
    int nx = q.size();
    qLeft[0] = q[0] + 0.5 * minmod(
        q[0] - q[nx-1], q[1] - q[0]
    );
    qLeft[nx-1] = q[nx-1] + 0.5 * minmod(
        q[nx-1] - q[nx-2], q[0] - q[nx-1]
    );
}



struct RightComponentFunctor {
    const thrust::device_vector<double>& q;
    MinMod minmod;

    RightComponentFunctor(const thrust::device_vector<double>& q) : q(q) {}

    __host__ __device__
    double operator()(const int& i) const {
        return q[i+1] - 0.5 * minmod(q[i+1] - q[i], q[i+2] - q[i+1]);
    }
};

void MUSCL::getRightComponent(
    const thrust::device_vector<double>& q, 
    thrust::device_vector<double>& qRight
)
{
    thrust::counting_iterator<int> indices(0);
    MinMod minmod;

    thrust::transform(
        indices, 
        indices + nx - 2, 
        qRight.begin(), 
        RightComponentFunctor(q)
    );

    //周期境界条件
    qRight[nx-2] = q[nx-1] - 0.5 * minmod(
        q[nx-1] - q[nx-2], q[0] - q[nx-1]
        );
    qRight[nx-1] = q[0] - 0.5 * minmod(
        q[0] - q[nx-1], q[1] - q[0]
        );
}

