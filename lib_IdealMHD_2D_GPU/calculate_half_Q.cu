#include "calculate_half_Q.hpp"


CalculateHalfQ::CalculateHalfQ()
    : dQCenter(nx * ny), 
      dQLeft(nx * ny), 
      dQRight(nx * ny)
{
}


struct GetBasicParamterFunctor {

    __device__
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

void CalculateHalfQ::setPhysicalParameters(
    const thrust::device_vector<ConservationParameter>& U
)
{
    thrust::transform(
        U.begin(), 
        U.end(),  
        dQCenter.begin(),
        GetBasicParamterFunctor()
    );
}


void CalculateHalfQ::calculateLeftQX()
{ 
    muscl.getLeftQX(dQCenter, dQLeft);
}


void CalculateHalfQ::calculateLeftQY()
{ 
    muscl.getLeftQY(dQCenter, dQLeft);
}


void CalculateHalfQ::calculateRightQX()
{ 
    muscl.getRightQX(dQCenter, dQRight);
}


void CalculateHalfQ::calculateRightQY()
{ 
    muscl.getRightQY(dQCenter, dQRight);
}



// getter

thrust::device_vector<BasicParameter> CalculateHalfQ::getCenterQ()
{
    return dQCenter;
}


thrust::device_vector<BasicParameter> CalculateHalfQ::getLeftQ()
{
    return dQLeft;
}


thrust::device_vector<BasicParameter> CalculateHalfQ::getRightQ()
{
    return dQRight;
}

