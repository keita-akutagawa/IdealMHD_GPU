#include "const.hpp"
#include "hlld.hpp"
#include <thrust/transform.h>
#include <thrust/tuple.h>


HLLD::HLLD(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo), 
      
      calculateHalfQ(mPIInfo), 
      
      dQCenter     (mPIInfo.localSizeX * mPIInfo.localSizeY),
      dQLeft       (mPIInfo.localSizeX * mPIInfo.localSizeY),
      dQRight      (mPIInfo.localSizeX * mPIInfo.localSizeY),
      hLLDParameter(mPIInfo.localSizeX * mPIInfo.localSizeY),

      flux           (mPIInfo.localSizeX * mPIInfo.localSizeY),
      fluxOuterLeft  (mPIInfo.localSizeX * mPIInfo.localSizeY),
      fluxMiddleLeft (mPIInfo.localSizeX * mPIInfo.localSizeY),
      fluxInnerLeft  (mPIInfo.localSizeX * mPIInfo.localSizeY),
      fluxOuterRight (mPIInfo.localSizeX * mPIInfo.localSizeY),
      fluxMiddleRight(mPIInfo.localSizeX * mPIInfo.localSizeY),
      fluxInnerRight (mPIInfo.localSizeX * mPIInfo.localSizeY), 

      tmpUForFluxG(mPIInfo.localSizeX * mPIInfo.localSizeY)
{
}


struct Sign {

    __device__
    double operator()(const double& x) const {
        return (x > 0.0) - (x < 0.0);
    }
};


struct CalculateFluxFunctor {

    __device__
    Flux operator()(
        const thrust::tuple<
        HLLDParameter, 
        Flux, Flux, Flux, 
        Flux, Flux, Flux
        >& tupleForFlux) const {
            HLLDParameter hLLDParameter = thrust::get<0>(tupleForFlux);
            Flux fluxOuterLeft          = thrust::get<1>(tupleForFlux);
            Flux fluxMiddleLeft         = thrust::get<2>(tupleForFlux);
            Flux fluxInnerLeft          = thrust::get<3>(tupleForFlux);
            Flux fluxOuterRight         = thrust::get<4>(tupleForFlux);
            Flux fluxMiddleRight        = thrust::get<5>(tupleForFlux);
            Flux fluxInnerRight         = thrust::get<6>(tupleForFlux);

            Flux flux;

            double SL = hLLDParameter.SL;
            double S1L = hLLDParameter.S1L;
            double SM = hLLDParameter.SM;
            double SR = hLLDParameter.SR;
            double S1R = hLLDParameter.S1R;

            auto calculateFluxComponent = [&](
                double outerLeftFlux, double middleLeftFlux, double innerLeftFlux,
                double outerRightFlux, double middleRightFlux, double innerRightFlux
            ) {
                return outerLeftFlux   * (SL > 0.0)
                     + middleLeftFlux  * ((SL <= 0.0) && (0.0 < S1L))
                     + innerLeftFlux   * ((S1L <= 0.0) && (0.0 < SM))
                     + outerRightFlux  * (SR <= 0.0)
                     + middleRightFlux * ((S1R <= 0.0) && (0.0 < SR))
                     + innerRightFlux  * ((SM <= 0.0) && (0.0 < S1R));
            };

            flux.f0 = calculateFluxComponent(
                fluxOuterLeft.f0, fluxMiddleLeft.f0, fluxInnerLeft.f0, 
                fluxOuterRight.f0, fluxMiddleRight.f0, fluxInnerRight.f0
            );

            flux.f1 = calculateFluxComponent(
                fluxOuterLeft.f1, fluxMiddleLeft.f1, fluxInnerLeft.f1,
                fluxOuterRight.f1, fluxMiddleRight.f1, fluxInnerRight.f1);

            flux.f2 = calculateFluxComponent(
                fluxOuterLeft.f2, fluxMiddleLeft.f2, fluxInnerLeft.f2,
                fluxOuterRight.f2, fluxMiddleRight.f2, fluxInnerRight.f2);

            flux.f3 = calculateFluxComponent(
                fluxOuterLeft.f3, fluxMiddleLeft.f3, fluxInnerLeft.f3,
                fluxOuterRight.f3, fluxMiddleRight.f3, fluxInnerRight.f3);

            flux.f4 = calculateFluxComponent(
                fluxOuterLeft.f4, fluxMiddleLeft.f4, fluxInnerLeft.f4,
                fluxOuterRight.f4, fluxMiddleRight.f4, fluxInnerRight.f4);

            flux.f5 = calculateFluxComponent(
                fluxOuterLeft.f5, fluxMiddleLeft.f5, fluxInnerLeft.f5,
                fluxOuterRight.f5, fluxMiddleRight.f5, fluxInnerRight.f5);

            flux.f6 = calculateFluxComponent(
                fluxOuterLeft.f6, fluxMiddleLeft.f6, fluxInnerLeft.f6,
                fluxOuterRight.f6, fluxMiddleRight.f6, fluxInnerRight.f6);

            flux.f7 = calculateFluxComponent(
                fluxOuterLeft.f7, fluxMiddleLeft.f7, fluxInnerLeft.f7,
                fluxOuterRight.f7, fluxMiddleRight.f7, fluxInnerRight.f7);

            return flux;
    }
};


void HLLD::calculateFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    setQX(U);
    calculateHLLDParameter();
    setFlux();

    auto tupleForFlux = thrust::make_tuple(
        hLLDParameter.begin(), 
        fluxOuterLeft.begin(), fluxMiddleLeft.begin(), fluxInnerLeft.begin(), 
        fluxOuterRight.begin(), fluxMiddleRight.begin(), fluxInnerRight.begin()
    );
    auto tupleForFluxIterator = thrust::make_zip_iterator(tupleForFlux);

    thrust::transform(
        tupleForFluxIterator, 
        tupleForFluxIterator + mPIInfo.localSizeX * mPIInfo.localSizeY, 
        flux.begin(), 
        CalculateFluxFunctor()
    );
}


__global__ void shuffleForTmpUForFluxG_kernel(
    const ConservationParameter* U, 
    ConservationParameter* tmpU, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < localSizeY) {
        tmpU[j + i * localSizeY].rho  = U[j + i * localSizeY].rho;
        tmpU[j + i * localSizeY].rhoU = U[j + i * localSizeY].rhoV;
        tmpU[j + i * localSizeY].rhoV = U[j + i * localSizeY].rhoW;
        tmpU[j + i * localSizeY].rhoW = U[j + i * localSizeY].rhoU;
        tmpU[j + i * localSizeY].bX   = U[j + i * localSizeY].bY;
        tmpU[j + i * localSizeY].bY   = U[j + i * localSizeY].bZ;
        tmpU[j + i * localSizeY].bZ   = U[j + i * localSizeY].bX;
        tmpU[j + i * localSizeY].e    = U[j + i * localSizeY].e;
    }
}

void HLLD::shuffleForTmpUForFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    shuffleForTmpUForFluxG_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(U.data()), 
        thrust::raw_pointer_cast(tmpUForFluxG.data()), 
        localSizeX, localSizeY
    );

    cudaDeviceSynchronize();
}


__global__ void shuffleForFluxG_kernel(
    Flux* flux, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    double f1, f2, f3, f4, f5, f6;

    if (i < localSizeX && j < localSizeY) {
        f1 = flux[j + i * localSizeY].f1;
        f2 = flux[j + i * localSizeY].f2;
        f3 = flux[j + i * localSizeY].f3;
        f4 = flux[j + i * localSizeY].f4;
        f5 = flux[j + i * localSizeY].f5;
        f6 = flux[j + i * localSizeY].f6;

        flux[j + i * localSizeY].f1 = f3;
        flux[j + i * localSizeY].f2 = f1;
        flux[j + i * localSizeY].f3 = f2;
        flux[j + i * localSizeY].f4 = f6;
        flux[j + i * localSizeY].f5 = f4;
        flux[j + i * localSizeY].f6 = f5;
    }
}

void HLLD::shuffleFluxG()
{
    int localSizeX = mPIInfo.localSizeX;
    int localSizeY = mPIInfo.localSizeY;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    shuffleForFluxG_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(flux.data()), 
        localSizeX, localSizeY
    );

    cudaDeviceSynchronize();
}


//fluxの計算はx方向のものを再利用するため、Uの入れ替えをする
void HLLD::calculateFluxG(
    const thrust::device_vector<ConservationParameter>& U
)
{
    shuffleForTmpUForFluxG(U);
    setQY(tmpUForFluxG);
    calculateHLLDParameter();
    setFlux();

    auto tupleForFlux = thrust::make_tuple(
        hLLDParameter.begin(), 
        fluxOuterLeft.begin(), fluxMiddleLeft.begin(), fluxInnerLeft.begin(), 
        fluxOuterRight.begin(), fluxMiddleRight.begin(), fluxInnerRight.begin()
    );
    auto tupleForFluxIterator = thrust::make_zip_iterator(tupleForFlux);

    thrust::transform(
        tupleForFluxIterator, 
        tupleForFluxIterator + mPIInfo.localSizeX * mPIInfo.localSizeY, 
        flux.begin(), 
        CalculateFluxFunctor()
    );

    shuffleFluxG();
}


void HLLD::setQX(
    const thrust::device_vector<ConservationParameter>& U
)
{
    calculateHalfQ.setPhysicalParameterX(U, dQCenter);
    calculateHalfQ.calculateLeftQX(dQCenter, dQLeft);
    calculateHalfQ.calculateRightQX(dQCenter, dQRight);
}


void HLLD::setQY(
    const thrust::device_vector<ConservationParameter>& U
)
{
    calculateHalfQ.setPhysicalParameterY(U, dQCenter);
    calculateHalfQ.calculateLeftQY(dQCenter, dQLeft);
    calculateHalfQ.calculateRightQY(dQCenter, dQRight);
}


struct calculateHLLDParameterFunctor {

    __device__
    HLLDParameter operator()(const BasicParameter& dQLeft,  const BasicParameter& dQRight) const {
        double rhoL, uL, vL, wL, bXL, bYL, bZL, eL, pL, pTL;
        double rhoR, uR, vR, wR, bXR, bYR, bZR, eR, pR, pTR;
        double csL, caL, vaL, cfL; 
        double csR, caR, vaR, cfR;
        double SL, SR, SM;
        double pT1, pT1L, pT1R;
        double rho1L, u1L, v1L, w1L, bX1L, bY1L, bZ1L, e1L;
        double rho1R, u1R, v1R, w1R, bX1R, bY1R, bZ1R, e1R;
        double S1L, S1R;
        double rho2L, rho2R, u2, v2, w2, bY2, bZ2, e2L, e2R, pT2L, pT2R;
        Sign sign;
        HLLDParameter hLLDParameter;

        rhoL = dQLeft.rho;
        uL   = dQLeft.u;
        vL   = dQLeft.v;
        wL   = dQLeft.w;
        bXL  = dQLeft.bX;
        bYL  = dQLeft.bY;
        bZL  = dQLeft.bZ;
        pL   = dQLeft.p;
        eL   = pL / (device_gamma_mhd - 1.0)
             + 0.5 * rhoL * (uL * uL + vL * vL + wL * wL)
             + 0.5 * (bXL * bXL + bYL * bYL + bZL * bZL); 
        pTL  = pL + 0.5 * (bXL * bXL + bYL * bYL + bZL * bZL);

        rhoR = dQRight.rho;
        uR   = dQRight.u;
        vR   = dQRight.v;
        wR   = dQRight.w;
        bXR  = dQRight.bX;
        bYR  = dQRight.bY;
        bZR  = dQRight.bZ;
        pR   = dQRight.p;
        eR   = pR / (device_gamma_mhd - 1.0)
             + 0.5 * rhoR * (uR * uR + vR * vR + wR * wR)
             + 0.5 * (bXR * bXR + bYR * bYR + bZR * bZR); 
        pTR  = pR + 0.5 * (bXR * bXR + bYR * bYR + bZR * bZR);


        csL = sqrt(device_gamma_mhd * pL / rhoL);
        caL = sqrt((bXL * bXL + bYL * bYL + bZL * bZL) / rhoL);
        vaL = sqrt(bXL * bXL / rhoL);
        cfL = sqrt(0.5 * (csL * csL + caL * caL
            + sqrt((csL * csL + caL * caL) * (csL * csL + caL * caL)
            - 4.0 * csL * csL * vaL * vaL)));
        
        csR = sqrt(device_gamma_mhd * pR / rhoR);
        caR = sqrt((bXR * bXR + bYR * bYR + bZR * bZR) / rhoR);
        vaR = sqrt(bXR * bXR / rhoR);
        cfR = sqrt(0.5 * (csR * csR + caR * caR
            + sqrt((csR * csR + caR * caR) * (csR * csR + caR * caR)
            - 4.0 * csR * csR * vaR * vaR)));
        

        SL = thrust::min(uL, uR) - thrust::max(cfL, cfR);
        SR = thrust::max(uL, uR) + thrust::max(cfL, cfR);
        SL = thrust::min(SL, 0.0);
        SR = thrust::max(SR, 0.0);

        SM = ((SR - uR) * rhoR * uR - (SL - uL) * rhoL * uL - pTR + pTL)
           / ((SR - uR) * rhoR - (SL - uL) * rhoL + device_EPS);

        pT1  = ((SR - uR) * rhoR * pTL - (SL - uL) * rhoL * pTR
             + rhoL * rhoR * (SR - uR) * (SL - uL) * (uR - uL))
             / ((SR - uR) * rhoR - (SL - uL) * rhoL + device_EPS);
        pT1L = pT1;
        pT1R = pT1;


        rho1L = rhoL * (SL - uL) / (SL - SM + device_EPS);
        u1L   = SM;
        v1L   = vL - bXL * bYL * (SM - uL) / (rhoL * (SL - uL) * (SL - SM) - bXL * bXL + device_EPS);
        w1L   = wL - bXL * bZL * (SM - uL) / (rhoL * (SL - uL) * (SL - SM) - bXL * bXL + device_EPS);
        bX1L  = bXL;
        bY1L  = bYL * (rhoL * (SL - uL) * (SL - uL) - bXL * bXL)
              / (rhoL * (SL - uL) * (SL - SM) - bXL * bXL + device_EPS);
        bZ1L  = bZL * (rhoL * (SL - uL) * (SL - uL) - bXL * bXL)
              / (rhoL * (SL - uL) * (SL - SM) - bXL * bXL + device_EPS);
        e1L   = ((SL - uL) * eL - pTL * uL + pT1 * SM
              + bXL * ((uL * bXL + vL * bYL + wL * bZL) - (u1L * bX1L + v1L * bY1L + w1L * bZ1L)))
              / (SL - SM + device_EPS);
        
        rho1R = rhoR * (SR - uR) / (SR - SM + device_EPS);
        u1R   = SM;
        v1R   = vR - bXR * bYR * (SM - uR) / (rhoR * (SR - uR) * (SR - SM) - bXR * bXR + device_EPS);
        w1R   = wR - bXR * bZR * (SM - uR) / (rhoR * (SR - uR) * (SR - SM) - bXR * bXR + device_EPS);
        bX1R  = bXR;
        bY1R  = bYR * (rhoR * (SR - uR) * (SR - uR) - bXR * bXR)
              / (rhoR * (SR - uR) * (SR - SM) - bXR * bXR + device_EPS);
        bZ1R  = bZR * (rhoR * (SR - uR) * (SR - uR) - bXR * bXR)
              / (rhoR * (SR - uR) * (SR - SM) - bXR * bXR + device_EPS);
        e1R   = ((SR - uR) * eR - pTR * uR + pT1 * SM
              + bXR * ((uR * bXR + vR * bYR + wR * bZR) - (u1R * bX1R + v1R * bY1R + w1R * bZ1R)))
              / (SR - SM + device_EPS);
        
        S1L = SM - sqrt(bX1L * bX1L / rho1L);
        S1R = SM + sqrt(bX1R * bX1R / rho1R);

        
        rho2L = rho1L;
        rho2R = rho1R;
        u2 = SM;
        v2 = (sqrt(rho1L) * v1L + sqrt(rho1R) * v1R + (bY1R - bY1L) * sign(bX1L))
            / (sqrt(rho1L) + sqrt(rho1R) + device_EPS);
        w2 = (sqrt(rho1L) * w1L + sqrt(rho1R) * w1R + (bZ1R - bZ1L) * sign(bX1L))
            / (sqrt(rho1L) + sqrt(rho1R) + device_EPS);
        bY2 = (sqrt(rho1L) * bY1R + sqrt(rho1R) * bY1L + sqrt(rho1L * rho1R) * (v1R - v1L) * sign(bX1L))
             / (sqrt(rho1L) + sqrt(rho1R) + device_EPS);
        bZ2 = (sqrt(rho1L) * bZ1R + sqrt(rho1R) * bZ1L + sqrt(rho1L * rho1R) * (w1R - w1L) * sign(bX1L))
             / (sqrt(rho1L) + sqrt(rho1R) + device_EPS);
        e2L = e1L - sqrt(rho1L)
            * ((u1L * bX1L + v1L * bY1L + w1L * bZ1L) - (u2 * bXL + v2 * bY2 + w2 * bZ2))
            * sign(bXL);
        e2R = e1R + sqrt(rho1R)
            * ((u1R * bX1R + v1R * bY1R + w1R * bZ1R) - (u2 * bXR + v2 * bY2 + w2 * bZ2))
            * sign(bXR);
        pT2L = pT1L;
        pT2R = pT1R;


        // set
        hLLDParameter.pTL = pTL;
        hLLDParameter.pTR = pTR;
        hLLDParameter.eL = eL;
        hLLDParameter.eR = eR;
        hLLDParameter.csL = csL;
        hLLDParameter.csR = csR;
        hLLDParameter.caL = caL;
        hLLDParameter.caR = caR;
        hLLDParameter.vaL = vaL;
        hLLDParameter.vaR = vaR;
        hLLDParameter.cfL = cfL;
        hLLDParameter.cfR = cfR;

        hLLDParameter.SL = SL;
        hLLDParameter.SR = SR;
        hLLDParameter.SM = SM;

        hLLDParameter.rho1L = rho1L;
        hLLDParameter.rho1R = rho1R;
        hLLDParameter.u1L = u1L;
        hLLDParameter.u1R = u1R;
        hLLDParameter.v1L = v1L;
        hLLDParameter.v1R = v1R;
        hLLDParameter.w1L = w1L;
        hLLDParameter.w1R = w1R;
        hLLDParameter.bY1L = bY1L;
        hLLDParameter.bY1R = bY1R;
        hLLDParameter.bZ1L = bZ1L;
        hLLDParameter.bZ1R = bZ1R;
        hLLDParameter.e1L = e1L;
        hLLDParameter.e1R = e1R;
        hLLDParameter.pT1L = pT1L;
        hLLDParameter.pT1R = pT1R;

        hLLDParameter.S1L = S1L;
        hLLDParameter.S1R = S1R;

        hLLDParameter.rho2L = rho2L;
        hLLDParameter.rho2R = rho2R;
        hLLDParameter.u2 = u2;
        hLLDParameter.v2 = v2;
        hLLDParameter.w2 = w2;
        hLLDParameter.bY2 = bY2;
        hLLDParameter.bZ2 = bZ2;
        hLLDParameter.e2L = e2L;
        hLLDParameter.e2R = e2R;
        hLLDParameter.pT2L = pT2L;
        hLLDParameter.pT2R = pT2R;

        return hLLDParameter;
    }
};

void HLLD::calculateHLLDParameter()
{
    thrust::transform(
        dQLeft.begin(), 
        dQLeft.end(), 
        dQRight.begin(), 
        hLLDParameter.begin(), 
        calculateHLLDParameterFunctor()
    );
}

///////////////////////////////////////////

__device__ 
Flux getOneFlux(
    double rho, double u, double v, double w, 
    double bX, double bY, double bZ, 
    double e, double pT
)
{
    Flux flux;

    flux.f0 = rho * u;
    flux.f1 = rho * u * u + pT - bX * bX;
    flux.f2 = rho * u * v - bX * bY;
    flux.f3 = rho * u * w - bX * bZ;
    flux.f4 = 0.0;
    flux.f5 = u * bY - v * bX;
    flux.f6 = u * bZ - w * bX;
    flux.f7 = (e + pT) * u - bX * (bX * u + bY * v + bZ * w);
    
    return flux;
};

struct setFluxFunctor {

    __device__
    thrust::tuple<Flux, Flux, Flux, Flux, Flux, Flux> operator()(
        const thrust::tuple<BasicParameter, BasicParameter, HLLDParameter>& tupleForFlux
    ) const {
        BasicParameter dQLeft       = thrust::get<0>(tupleForFlux);
        BasicParameter dQRight      = thrust::get<1>(tupleForFlux);
        HLLDParameter hLLDParameter = thrust::get<2>(tupleForFlux);

        double rhoL, uL, vL, wL, bXL, bYL, bZL, eL, pTL;
        double rhoR, uR, vR, wR, bXR, bYR, bZR, eR, pTR;
        double rho1L, u1L, v1L, w1L, bY1L, bZ1L, e1L, pT1L;
        double rho1R, u1R, v1R, w1R, bY1R, bZ1R, e1R, pT1R;
        double rho2L, rho2R, u2, v2, w2, bY2, bZ2, e2L, e2R, pT2L, pT2R;
        Flux fluxOuterLeft, fluxMiddleLeft, fluxInnerLeft;
        Flux fluxOuterRight, fluxMiddleRight, fluxInnerRight;
    
    
        rhoL = dQLeft.rho;
        uL   = dQLeft.u;
        vL   = dQLeft.v;
        wL   = dQLeft.w;
        bXL  = dQLeft.bX;
        bYL  = dQLeft.bY;
        bZL  = dQLeft.bZ;
        eL   = hLLDParameter.eL;
        pTL  = hLLDParameter.pTL;

        rhoR = dQRight.rho;
        uR   = dQRight.u;
        vR   = dQRight.v;
        wR   = dQRight.w;
        bXR  = dQRight.bX;
        bYR  = dQRight.bY;
        bZR  = dQRight.bZ;
        eR   = hLLDParameter.eR;
        pTR  = hLLDParameter.pTR;

        rho1L = hLLDParameter.rho1L;
        rho1R = hLLDParameter.rho1R;
        u1L   = hLLDParameter.u1L;
        u1R   = hLLDParameter.u1R;
        v1L   = hLLDParameter.v1L;
        v1R   = hLLDParameter.v1R;
        w1L   = hLLDParameter.w1L;
        w1R   = hLLDParameter.w1R;
        bY1L  = hLLDParameter.bY1L;
        bY1R  = hLLDParameter.bY1R;
        bZ1L  = hLLDParameter.bZ1L;
        bZ1R  = hLLDParameter.bZ1R;
        e1L   = hLLDParameter.e1L;
        e1R   = hLLDParameter.e1R;
        pT1L  = hLLDParameter.pT1L;
        pT1R  = hLLDParameter.pT1R;

        rho2L = hLLDParameter.rho2L;
        rho2R = hLLDParameter.rho2R;
        u2    = hLLDParameter.u2;
        v2    = hLLDParameter.v2;
        w2    = hLLDParameter.w2;
        bY2   = hLLDParameter.bY2;
        bZ2   = hLLDParameter.bZ2;
        e2L   = hLLDParameter.e2L;
        e2R   = hLLDParameter.e2R;
        pT2L  = hLLDParameter.pT2L;
        pT2R  = hLLDParameter.pT2R;

        fluxOuterLeft   = getOneFlux(rhoL, uL, vL, wL, bXL, bYL, bZL, eL, pTL);
        fluxMiddleLeft  = getOneFlux(rho1L, u1L, v1L, w1L, bXL, bY1L, bZ1L, e1L, pT1L);
        fluxInnerLeft   = getOneFlux(rho2L, u2, v2, w2, bXL, bY2, bZ2, e2L, pT2L);
        fluxOuterRight  = getOneFlux(rhoR, uR, vR, wR, bXR, bYR, bZR, eR, pTR);
        fluxMiddleRight = getOneFlux(rho1R, u1R, v1R, w1R, bXR, bY1R, bZ1R, e1R, pT1R);
        fluxInnerRight  = getOneFlux(rho2R, u2, v2, w2, bXR, bY2, bZ2, e2R, pT2R);

        return thrust::make_tuple(
            fluxOuterLeft, fluxMiddleLeft, fluxInnerLeft, 
            fluxOuterRight, fluxMiddleRight, fluxInnerRight
        );
    }
};

void HLLD::setFlux()
{
    auto tupleForFlux = thrust::make_tuple(dQLeft.begin(), dQRight.begin(), hLLDParameter.begin());
    auto tupleForFluxIterator = thrust::make_zip_iterator(tupleForFlux);

    thrust::transform(
        tupleForFluxIterator, 
        tupleForFluxIterator + mPIInfo.localSizeX * mPIInfo.localSizeY, 
        thrust::make_zip_iterator(
            thrust::make_tuple(fluxOuterLeft.begin(), fluxMiddleLeft.begin(), fluxInnerLeft.begin(), 
                               fluxOuterRight.begin(), fluxMiddleRight.begin(), fluxInnerRight.begin())
        ),
        setFluxFunctor()
    );
}


//getter

thrust::device_vector<Flux>& HLLD::getFlux()
{
    return flux;
}
