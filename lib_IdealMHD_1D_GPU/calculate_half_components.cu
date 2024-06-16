#include "calculate_half_components.hpp"


struct GetBasicParamterFunctor {
    MinMod minmod;

    __host__ __device__
    BasicParameter operator()(const ConservationParameter& conservationParameter) const {
        BasicParameter dQ;        
        double rho, u, v, w, bX, bY, bZ, e, p;

        rho = conservationParameter.rho;
        u = conservationParameter.rhoU / rho;
        v = conservationParameter.rhoV / rho;
        w = conservationParameter.rhoW / rho;
        bX = conservationParameter.bX;
        bY = conservationParameter.bY;
        bZ = conservationParameter.bZ;
        e = conservationParameter.e;
        p = (device_gamma_mhd - 1.0)
          * (e - 0.5 * (rho * (u * u + v * v + w * w))
          - 0.5 * (bX * bX + bY * bY + bZ * bZ));
        
        dQ.rho = rho;
        dQ.u = u;
        dQ.v = v;
        dQ.w = w;
        dQ.bX = bX;
        dQ.bY = bY;
        dQ.bZ = bZ;
        dQ.p = p;

        return dQ;
    }
};

void CalculateHalfComponents::setPhysicalParameters(
    const thrust::device_vector<ConservationParameter>& U
)
{
    auto tupleOfBasicParamter = thrust::make_tuple(U.begin());
    auto tupleOfBasicParameterIterator = thrust::make_zip_iterator(tupleOfBasicParamter);

    thrust::transform(
        tupleOfBasicParameterIterator, 
        tupleOfBasicParameterIterator + nx, 
        dQCenter.begin(),
        GetBasicParamterFunctor()
    );
}


void CalculateHalfComponents::calculateLeftComponents()
{ 
    muscl.getLeftComponent(dQCenter, dQLeft);
}


void CalculateHalfComponents::calculateRightComponents()
{ 
    muscl.getRightComponent(dQCenter, dQRight);
}


thrust::device_vector<BasicParameter> CalculateHalfComponents::getCenterComponents()
{
    return dQCenter;
}


thrust::device_vector<BasicParameter> CalculateHalfComponents::getLeftComponents()
{
    return dQLeft;
}


thrust::device_vector<BasicParameter> CalculateHalfComponents::getRightComponents()
{
    return dQRight;
}

