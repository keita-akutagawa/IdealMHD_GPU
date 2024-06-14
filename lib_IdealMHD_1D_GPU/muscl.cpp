#include "muscl.hpp"
#include <thrust/transform.h>


struct LeftComponentFunctor {
    const thrust::device_vector<double>& q;

    LeftComponentFunctor(const thrust::device_vector<double>& q) : q(q) {}

    __host__ __device__
    double operator()(const int& i) const {
        return q[i] + 0.5 * minmod(q[i] - q[i-1], q[i+1] - q[i]);
    }
};

void MUSCL::getLeftComponent(
    const thrust::device_vector<double> q, 
    thrust::device_vector<double>& qLeft
)
{
    thrust::counting_iterator<int> indices(0);

    thrust::transform(
        indices + 1, 
        indices + nx - 1, 
        qLeft.begin() + 1, 
        LeftComponentFunctor(q)
    );

    //周期境界条件
    qLeft[0] = q[0] + 0.5 * minmod(
        q[0] - q[nx-1], q[1] - q[0]
        );
    qLeft[nx-1] = q[nx-1] + 0.5 * minmod(
        q[nx-1] - q[nx-2], q[0] - q[nx-1]
        );
}


struct RightComponentFunctor {
    const thrust::device_vector<double>& q;

    LeftComponentFunctor(const thrust::device_vector<double>& q) : q(q) {}

    __host__ __device__
    double operator()(const int& i) const {
        return q[i+1] - 0.5 * minmod(q[i+1] - q[i], q[i+2] - q[i+1]);
    }
};

void MUSCL::getRightComponent(
    const thrust::device_vector<double> q, 
    thrust::device_vector<double>& qRight
)
{
    thrust::counting_iterator<int> indices(0);

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

