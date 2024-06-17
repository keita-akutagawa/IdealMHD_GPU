#include "muscl.hpp"
#include <thrust/transform.h>
#include <thrust/tuple.h>


struct LeftParameterFunctor {
    MinMod minmod;

    __device__
    BasicParameter operator()(const thrust::tuple<BasicParameter, BasicParameter, BasicParameter>& tupleForLeft) const {
        BasicParameter dQMinus1 = thrust::get<0>(tupleForLeft);
        BasicParameter dQ = thrust::get<1>(tupleForLeft);
        BasicParameter dQPlus1 = thrust::get<2>(tupleForLeft);

        BasicParameter dQLeft;

        dQLeft.rho = dQ.rho + 0.5 * minmod(dQ.rho - dQMinus1.rho, dQPlus1.rho - dQ.rho);
        dQLeft.u   = dQ.u   + 0.5 * minmod(dQ.u   - dQMinus1.u  , dQPlus1.u   - dQ.u  );
        dQLeft.v   = dQ.v   + 0.5 * minmod(dQ.v   - dQMinus1.v  , dQPlus1.v   - dQ.v  );
        dQLeft.w   = dQ.w   + 0.5 * minmod(dQ.w   - dQMinus1.w  , dQPlus1.w   - dQ.w  );
        dQLeft.bX  = dQ.bX  + 0.5 * minmod(dQ.bX  - dQMinus1.bX , dQPlus1.bX  - dQ.bX );
        dQLeft.bY  = dQ.bY  + 0.5 * minmod(dQ.bY  - dQMinus1.bY , dQPlus1.bY  - dQ.bY );
        dQLeft.bZ  = dQ.bZ  + 0.5 * minmod(dQ.bZ  - dQMinus1.bZ , dQPlus1.bZ  - dQ.bZ );
        dQLeft.p   = dQ.p   + 0.5 * minmod(dQ.p   - dQMinus1.p  , dQPlus1.p   - dQ.p  );

        return dQLeft;
    }
};


void MUSCL::getLeftComponent(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{
    auto tupleForLeft = thrust::make_tuple(dQ.begin() - 1, dQ.begin(), dQ.begin() + 1);
    auto tupleForLeftIterator = thrust::make_zip_iterator(tupleForLeft);

    thrust::transform(
        tupleForLeftIterator + 1, 
        tupleForLeftIterator + nx - 1, 
        dQLeft.begin() + 1,
        LeftParameterFunctor()
    );
}



struct RightParameterFunctor {
    MinMod minmod;

    __device__
    BasicParameter operator()(const thrust::tuple<BasicParameter, BasicParameter, BasicParameter>& tupleForRight) const {
        BasicParameter dQ = thrust::get<0>(tupleForRight);
        BasicParameter dQPlus1 = thrust::get<1>(tupleForRight);
        BasicParameter dQPlus2 = thrust::get<2>(tupleForRight);

        BasicParameter dQRight;

        dQRight.rho = dQPlus1.rho - 0.5 * minmod(dQPlus1.rho - dQ.rho, dQPlus2.rho - dQPlus1.rho);
        dQRight.u   = dQPlus1.u   - 0.5 * minmod(dQPlus1.u   - dQ.u  , dQPlus2.u   - dQPlus1.u);
        dQRight.v   = dQPlus1.v   - 0.5 * minmod(dQPlus1.v   - dQ.v  , dQPlus2.v   - dQPlus1.v);
        dQRight.w   = dQPlus1.w   - 0.5 * minmod(dQPlus1.w   - dQ.w  , dQPlus2.w   - dQPlus1.w);
        dQRight.bX  = dQPlus1.bX  - 0.5 * minmod(dQPlus1.bX  - dQ.bX , dQPlus2.bX  - dQPlus1.bX);
        dQRight.bY  = dQPlus1.bY  - 0.5 * minmod(dQPlus1.bY  - dQ.bY , dQPlus2.bY  - dQPlus1.bY);
        dQRight.bZ  = dQPlus1.bZ  - 0.5 * minmod(dQPlus1.bZ  - dQ.bZ , dQPlus2.bZ  - dQPlus1.bZ);
        dQRight.p   = dQPlus1.p   - 0.5 * minmod(dQPlus1.p   - dQ.p  , dQPlus2.p   - dQPlus1.p);

        return dQRight;
    }
};


void MUSCL::getRightComponent(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQRight
)
{
    //thrust::tuple<thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator, thrust::device_vector<double>::iterator>
    auto tupleForRight = thrust::make_tuple(dQ.begin(), dQ.begin() + 1, dQ.begin() + 2);
    auto tupleForRightIterator = thrust::make_zip_iterator(tupleForRight);

    thrust::transform(
        tupleForRightIterator, 
        tupleForRightIterator + nx - 2, 
        dQRight.begin(),
        RightParameterFunctor()
    );
}

