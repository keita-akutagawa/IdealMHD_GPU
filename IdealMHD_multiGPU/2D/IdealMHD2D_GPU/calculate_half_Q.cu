#include "calculate_half_Q.hpp"


CalculateHalfQ::CalculateHalfQ(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      muscl(mPIInfo)
{
}


struct GetBasicParamterFunctor {

    __device__
    BasicParameter operator()(
        const ConservationParameter& conservationParameter, 
        const ConservationParameter& conservationParameterPlus1) const {
        BasicParameter dQ;        
        double rho, u, v, w, bX, bY, bZ, e, p;
        double bXPlus1;

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
        
        bXPlus1 = conservationParameterPlus1.bX;
        
        dQ.rho = rho;
        dQ.u = u;
        dQ.v = v;
        dQ.w = w;
        dQ.bX = 0.5 * (bX + bXPlus1);
        dQ.bY = bY;
        dQ.bZ = bZ;
        dQ.p = p;

        return dQ;
    }
};

void CalculateHalfQ::setPhysicalParameterX(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<BasicParameter>& dQCenter
)
{
    thrust::transform(
        U.begin(), 
        U.end() - mPIInfo.localSizeY,  
        U.begin() + mPIInfo.localSizeY,
        dQCenter.begin(),
        GetBasicParamterFunctor()
    );
}

void CalculateHalfQ::setPhysicalParameterY(
    const thrust::device_vector<ConservationParameter>& U, 
    thrust::device_vector<BasicParameter>& dQCenter
)
{
    thrust::transform(
        U.begin(), 
        U.end() - 1,  
        U.begin() + 1,
        dQCenter.begin(),
        GetBasicParamterFunctor()
    );
}


void CalculateHalfQ::calculateLeftQX(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{ 
    muscl.getLeftQX(dQCenter, dQLeft);
}


void CalculateHalfQ::calculateLeftQY(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{ 
    muscl.getLeftQY(dQCenter, dQLeft);
}


void CalculateHalfQ::calculateRightQX(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQRight
)
{ 
    muscl.getRightQX(dQCenter, dQRight);
}


void CalculateHalfQ::calculateRightQY(
    const thrust::device_vector<BasicParameter>& dQCenter, 
    thrust::device_vector<BasicParameter>& dQRight
)
{ 
    muscl.getRightQY(dQCenter, dQRight);
}


