#include "muscl.hpp"
#include <thrust/adjacent_difference.h>
#include <thrust/transform.h>
#include<thrust/tuple.h>


struct LeftComponentFunctor {
    MinMod minmod;

    __host__ __device__
    double operator()(const thrust::tuple<double, double, double>& tupleForLeft) const {
        double qMinus1 = thrust::get<0>(tupleForLeft);
        double q = thrust::get<1>(tupleForLeft);
        double qPlus1 = thrust::get<2>(tupleForLeft);

        return q + 0.5 * minmod(q - qMinus1, qPlus1 - q);
    }
};


void MUSCL::getLeftComponent(
    const thrust::device_vector<double>& q, 
    thrust::device_vector<double>& qLeft
)
{
    //thrust::tuple<thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator>
    auto tupleForLeft = thrust::make_tuple(q.begin() - 1, q.begin(), q.begin() + 1);
    auto tupleForLeftIterator = thrust::make_zip_iterator(tupleForLeft);

    thrust::transform(
        tupleForLeftIterator + 1, 
        tupleForLeftIterator + nx - 1, 
        qLeft.begin() + 1,
        LeftComponentFunctor()
    );

    MinMod minmod;
    // 周期境界条件
    qLeft[0] = q[0] + 0.5 * minmod(
        q[0] - q[nx-1], q[1] - q[0]
    );
    qLeft[nx-1] = q[nx-1] + 0.5 * minmod(
        q[nx-1] - q[nx-2], q[0] - q[nx-1]
    );
}



struct RightComponentFunctor {
    MinMod minmod;

    __host__ __device__
    double operator()(const thrust::tuple<double, double, double>& tupleForRight) const {
        double q = thrust::get<0>(tupleForRight);
        double qPlus1 = thrust::get<1>(tupleForRight);
        double qPlus2 = thrust::get<2>(tupleForRight);

        return qPlus1 - 0.5 * minmod(qPlus1 - q, qPlus2 - qPlus1);
    }
};

void MUSCL::getRightComponent(
    const thrust::device_vector<double>& q, 
    thrust::device_vector<double>& qRight
)
{
    //thrust::tuple<thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator>
    auto tupleForRight = thrust::make_tuple(q.begin(), q.begin() + 1, q.begin() + 2);
    auto tupleForRightIterator = thrust::make_zip_iterator(tupleForRight);

    thrust::transform(
        tupleForRightIterator, 
        tupleForRightIterator + nx - 2, 
        qRight.begin(),
        RightComponentFunctor()
    );

    MinMod minmod;
    //周期境界条件
    qRight[nx-2] = q[nx-1] - 0.5 * minmod(
        q[nx-1] - q[nx-2], q[0] - q[nx-1]
        );
    qRight[nx-1] = q[0] - 0.5 * minmod(
        q[0] - q[nx-1], q[1] - q[0]
        );
}

