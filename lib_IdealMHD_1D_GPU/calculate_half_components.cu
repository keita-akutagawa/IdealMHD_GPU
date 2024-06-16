#include "calculate_half_components.hpp"






void CalculateHalfComponents::setPhysicalParameters(
    const thrust::device_vector<ConservationParameter>& U
)
{
    auto tupleForLeft = thrust::make_tuple(U.begin());
    auto tupleForLeftIterator = thrust::make_zip_iterator(tupleForLeft);

    
}


void CalculateHalfComponents::calculateLeftComponents()
{ 
    muscl.getLeftComponent(componentsCenter.rho, componentsLeft.rho);
    muscl.getLeftComponent(componentsCenter.u,   componentsLeft.u);
    muscl.getLeftComponent(componentsCenter.v,   componentsLeft.v);
    muscl.getLeftComponent(componentsCenter.w,   componentsLeft.w);
    muscl.getLeftComponent(componentsCenter.by,  componentsLeft.by);
    muscl.getLeftComponent(componentsCenter.bz,  componentsLeft.bz);
    muscl.getLeftComponent(componentsCenter.p,   componentsLeft.p);

    for (int i = 0; i < nx; i++) {
        componentsLeft.bx[i] = componentsCenter.bx[i];
    }
}


void CalculateHalfComponents::calculateRightComponents()
{ 
    muscl.getRightComponent(componentsCenter.rho, componentsRight.rho);
    muscl.getRightComponent(componentsCenter.u,   componentsRight.u);
    muscl.getRightComponent(componentsCenter.v,   componentsRight.v);
    muscl.getRightComponent(componentsCenter.w,   componentsRight.w);
    muscl.getRightComponent(componentsCenter.by,  componentsRight.by);
    muscl.getRightComponent(componentsCenter.bz,  componentsRight.bz);
    muscl.getRightComponent(componentsCenter.p,   componentsRight.p);

    for (int i = 0; i < nx; i++) {
        componentsRight.bx[i] = componentsCenter.bx[i];
    }
}


thrust::device_vector<BasicParameter> CalculateHalfComponents::getCenterComponents()
{
    return qCenter;
}


thrust::device_vector<BasicParameter> CalculateHalfComponents::getLeftComponents()
{
    return qLeft;
}


thrust::device_vector<BasicParameter> CalculateHalfComponents::getRightComponents()
{
    return qRight;
}

