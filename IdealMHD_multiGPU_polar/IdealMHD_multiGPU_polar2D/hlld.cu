#include "hlld.hpp"


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

__device__ 
inline double calculateFluxComponent(
    double fluxOuterLeft, double fluxMiddleLeft, double fluxInnerLeft,
    double fluxOuterRight, double fluxMiddleRight, double fluxInnerRight, 
    double SL, double S1L, double SM, double SR, double S1R
)
{
    return fluxOuterLeft   * (SL > 0.0)
         + fluxMiddleLeft  * ((SL <= 0.0) && (0.0 < S1L))
         + fluxInnerLeft   * ((S1L <= 0.0) && (0.0 < SM))
         + fluxOuterRight  * (SR <= 0.0)
         + fluxMiddleRight * ((S1R <= 0.0) && (0.0 < SR))
         + fluxInnerRight  * ((SM <= 0.0) && (0.0 < S1R));
}

__global__ void calculateFlux_kernel(
    const HLLDParameter* hLLDParameter, 
    const Flux* fluxOuterLeft, const Flux* fluxMiddleLeft, const Flux* fluxInnerLeft, 
    const Flux* fluxOuterRight, const Flux* fluxMiddleRight, const Flux* fluxInnerRight, 
    Flux* flux, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < localSizeY) {
        int index = j + i * localSizeY;

        double SL  = hLLDParameter[index].SL;
        double S1L = hLLDParameter[index].S1L;
        double SM  = hLLDParameter[index].SM;
        double SR  = hLLDParameter[index].SR;
        double S1R = hLLDParameter[index].S1R;

        flux[index].f0 = calculateFluxComponent(
            fluxOuterLeft[index].f0, fluxMiddleLeft[index].f0, fluxInnerLeft[index].f0, 
            fluxOuterRight[index].f0, fluxMiddleRight[index].f0, fluxInnerRight[index].f0, 
            SL, S1L, SM, SR, S1R
        );
        flux[index].f1 = calculateFluxComponent(
            fluxOuterLeft[index].f1, fluxMiddleLeft[index].f1, fluxInnerLeft[index].f1, 
            fluxOuterRight[index].f1, fluxMiddleRight[index].f1, fluxInnerRight[index].f1, 
            SL, S1L, SM, SR, S1R
        );
        flux[index].f2 = calculateFluxComponent(
            fluxOuterLeft[index].f2, fluxMiddleLeft[index].f2, fluxInnerLeft[index].f2, 
            fluxOuterRight[index].f2, fluxMiddleRight[index].f2, fluxInnerRight[index].f2, 
            SL, S1L, SM, SR, S1R
        );
        flux[index].f3 = calculateFluxComponent(
            fluxOuterLeft[index].f3, fluxMiddleLeft[index].f3, fluxInnerLeft[index].f3, 
            fluxOuterRight[index].f3, fluxMiddleRight[index].f3, fluxInnerRight[index].f3, 
            SL, S1L, SM, SR, S1R
        );
        flux[index].f4 = calculateFluxComponent(
            fluxOuterLeft[index].f4, fluxMiddleLeft[index].f4, fluxInnerLeft[index].f4, 
            fluxOuterRight[index].f4, fluxMiddleRight[index].f4, fluxInnerRight[index].f4, 
            SL, S1L, SM, SR, S1R
        );
        flux[index].f5 = calculateFluxComponent(
            fluxOuterLeft[index].f5, fluxMiddleLeft[index].f5, fluxInnerLeft[index].f5, 
            fluxOuterRight[index].f5, fluxMiddleRight[index].f5, fluxInnerRight[index].f5, 
            SL, S1L, SM, SR, S1R
        );
        flux[index].f6 = calculateFluxComponent(
            fluxOuterLeft[index].f6, fluxMiddleLeft[index].f6, fluxInnerLeft[index].f6, 
            fluxOuterRight[index].f6, fluxMiddleRight[index].f6, fluxInnerRight[index].f6, 
            SL, S1L, SM, SR, S1R
        );
        flux[index].f7 = calculateFluxComponent(
            fluxOuterLeft[index].f7, fluxMiddleLeft[index].f7, fluxInnerLeft[index].f7, 
            fluxOuterRight[index].f7, fluxMiddleRight[index].f7, fluxInnerRight[index].f7, 
            SL, S1L, SM, SR, S1R
        );
    }
}


void HLLD::calculateFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    setQX(U);
    calculateHLLDParameter();
    setFlux();

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    calculateFlux_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(hLLDParameter.data()), 
        thrust::raw_pointer_cast(fluxOuterLeft.data()), 
        thrust::raw_pointer_cast(fluxMiddleLeft.data()), 
        thrust::raw_pointer_cast(fluxInnerLeft.data()), 
        thrust::raw_pointer_cast(fluxOuterRight.data()), 
        thrust::raw_pointer_cast(fluxMiddleRight.data()), 
        thrust::raw_pointer_cast(fluxInnerRight.data()), 
        thrust::raw_pointer_cast(flux.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
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
        int index = j + i * localSizeY;

        tmpU[index].rho  = U[index].rho;
        tmpU[index].rhoU = U[index].rhoV;
        tmpU[index].rhoV = U[index].rhoW;
        tmpU[index].rhoW = U[index].rhoU;
        tmpU[index].bX   = U[index].bY;
        tmpU[index].bY   = U[index].bZ;
        tmpU[index].bZ   = U[index].bX;
        tmpU[index].e    = U[index].e;
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

    if (i < localSizeX && j < localSizeY) {
        int index = j + i * localSizeY;
        double f1, f2, f3, f4, f5, f6;

        f1 = flux[index].f1;
        f2 = flux[index].f2;
        f3 = flux[index].f3;
        f4 = flux[index].f4;
        f5 = flux[index].f5;
        f6 = flux[index].f6;

        flux[index].f1 = f3;
        flux[index].f2 = f1;
        flux[index].f3 = f2;
        flux[index].f4 = f6;
        flux[index].f5 = f4;
        flux[index].f6 = f5;
    }
}

void HLLD::shuffleFluxG()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    shuffleForFluxG_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(flux.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
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

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    calculateFlux_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(hLLDParameter.data()), 
        thrust::raw_pointer_cast(fluxOuterLeft.data()), 
        thrust::raw_pointer_cast(fluxMiddleLeft.data()), 
        thrust::raw_pointer_cast(fluxInnerLeft.data()), 
        thrust::raw_pointer_cast(fluxOuterRight.data()), 
        thrust::raw_pointer_cast(fluxMiddleRight.data()), 
        thrust::raw_pointer_cast(fluxInnerRight.data()), 
        thrust::raw_pointer_cast(flux.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
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


__global__ void calculateHLLDParameter_kernel(
    const BasicParameter* dQLeft, const BasicParameter* dQRight, 
    HLLDParameter* hLLDParameter, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < localSizeY) {
        int index = j + i * localSizeY;

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

        rhoL = dQLeft[index].rho;
        uL   = dQLeft[index].u;
        vL   = dQLeft[index].v;
        wL   = dQLeft[index].w;
        bXL  = dQLeft[index].bX;
        bYL  = dQLeft[index].bY;
        bZL  = dQLeft[index].bZ;
        pL   = dQLeft[index].p;
        eL   = pL / (device_gamma_mhd - 1.0)
             + 0.5 * rhoL * (uL * uL + vL * vL + wL * wL)
             + 0.5 * (bXL * bXL + bYL * bYL + bZL * bZL); 
        pTL  = pL + 0.5 * (bXL * bXL + bYL * bYL + bZL * bZL);

        rhoR = dQRight[index].rho;
        uR   = dQRight[index].u;
        vR   = dQRight[index].v;
        wR   = dQRight[index].w;
        bXR  = dQRight[index].bX;
        bYR  = dQRight[index].bY;
        bZR  = dQRight[index].bZ;
        pR   = dQRight[index].p;
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


        hLLDParameter[index].pTL = pTL;
        hLLDParameter[index].pTR = pTR;
        hLLDParameter[index].eL  = eL;
        hLLDParameter[index].eR  = eR;
        hLLDParameter[index].csL = csL;
        hLLDParameter[index].csR = csR;
        hLLDParameter[index].caL = caL;
        hLLDParameter[index].caR = caR;
        hLLDParameter[index].vaL = vaL;
        hLLDParameter[index].vaR = vaR;
        hLLDParameter[index].cfL = cfL;
        hLLDParameter[index].cfR = cfR;

        hLLDParameter[index].SL = SL;
        hLLDParameter[index].SR = SR;
        hLLDParameter[index].SM = SM;

        hLLDParameter[index].rho1L = rho1L;
        hLLDParameter[index].rho1R = rho1R;
        hLLDParameter[index].u1L   = u1L;
        hLLDParameter[index].u1R   = u1R;
        hLLDParameter[index].v1L   = v1L;
        hLLDParameter[index].v1R   = v1R;
        hLLDParameter[index].w1L   = w1L;
        hLLDParameter[index].w1R   = w1R;
        hLLDParameter[index].bY1L  = bY1L;
        hLLDParameter[index].bY1R  = bY1R;
        hLLDParameter[index].bZ1L  = bZ1L;
        hLLDParameter[index].bZ1R  = bZ1R;
        hLLDParameter[index].e1L   = e1L;
        hLLDParameter[index].e1R   = e1R;
        hLLDParameter[index].pT1L  = pT1L;
        hLLDParameter[index].pT1R  = pT1R;

        hLLDParameter[index].S1L = S1L;
        hLLDParameter[index].S1R = S1R;

        hLLDParameter[index].rho2L = rho2L;
        hLLDParameter[index].rho2R = rho2R;
        hLLDParameter[index].u2    = u2;
        hLLDParameter[index].v2    = v2;
        hLLDParameter[index].w2    = w2;
        hLLDParameter[index].bY2   = bY2;
        hLLDParameter[index].bZ2   = bZ2;
        hLLDParameter[index].e2L   = e2L;
        hLLDParameter[index].e2R   = e2R;
        hLLDParameter[index].pT2L  = pT2L;
        hLLDParameter[index].pT2R  = pT2R;
    
    }
}
  

void HLLD::calculateHLLDParameter()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    calculateHLLDParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQLeft.data()), 
        thrust::raw_pointer_cast(dQRight.data()), 
        thrust::raw_pointer_cast(hLLDParameter.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    );
}

///////////////////////////////////////////

__device__ 
inline Flux getOneFlux(
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

__global__ void setFlux_kernel(
    const BasicParameter* dQLeft, const BasicParameter* dQRight, const HLLDParameter* hLLDParameter, 
    Flux* fluxOuterLeft, Flux* fluxMiddleLeft, Flux* fluxInnerLeft, 
    Flux* fluxOuterRight, Flux* fluxMiddleRight, Flux* fluxInnerRight, 
    int localSizeX, int localSizeY
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX && j < localSizeY) {
        int index = j + i * localSizeY;

        double rhoL, uL, vL, wL, bXL, bYL, bZL, eL, pTL;
        double rhoR, uR, vR, wR, bXR, bYR, bZR, eR, pTR;
        double rho1L, u1L, v1L, w1L, bY1L, bZ1L, e1L, pT1L;
        double rho1R, u1R, v1R, w1R, bY1R, bZ1R, e1R, pT1R;
        double rho2L, rho2R, u2, v2, w2, bY2, bZ2, e2L, e2R, pT2L, pT2R;

        rhoL = dQLeft[index].rho;
        uL   = dQLeft[index].u;
        vL   = dQLeft[index].v;
        wL   = dQLeft[index].w;
        bXL  = dQLeft[index].bX;
        bYL  = dQLeft[index].bY;
        bZL  = dQLeft[index].bZ;
        eL   = hLLDParameter[index].eL;
        pTL  = hLLDParameter[index].pTL;

        rhoR = dQRight[index].rho;
        uR   = dQRight[index].u;
        vR   = dQRight[index].v;
        wR   = dQRight[index].w;
        bXR  = dQRight[index].bX;
        bYR  = dQRight[index].bY;
        bZR  = dQRight[index].bZ;
        eR   = hLLDParameter[index].eR;
        pTR  = hLLDParameter[index].pTR;

        rho1L = hLLDParameter[index].rho1L;
        rho1R = hLLDParameter[index].rho1R;
        u1L   = hLLDParameter[index].u1L;
        u1R   = hLLDParameter[index].u1R;
        v1L   = hLLDParameter[index].v1L;
        v1R   = hLLDParameter[index].v1R;
        w1L   = hLLDParameter[index].w1L;
        w1R   = hLLDParameter[index].w1R;
        bY1L  = hLLDParameter[index].bY1L;
        bY1R  = hLLDParameter[index].bY1R;
        bZ1L  = hLLDParameter[index].bZ1L;
        bZ1R  = hLLDParameter[index].bZ1R;
        e1L   = hLLDParameter[index].e1L;
        e1R   = hLLDParameter[index].e1R;
        pT1L  = hLLDParameter[index].pT1L;
        pT1R  = hLLDParameter[index].pT1R;

        rho2L = hLLDParameter[index].rho2L;
        rho2R = hLLDParameter[index].rho2R;
        u2    = hLLDParameter[index].u2;
        v2    = hLLDParameter[index].v2;
        w2    = hLLDParameter[index].w2;
        bY2   = hLLDParameter[index].bY2;
        bZ2   = hLLDParameter[index].bZ2;
        e2L   = hLLDParameter[index].e2L;
        e2R   = hLLDParameter[index].e2R;
        pT2L  = hLLDParameter[index].pT2L;
        pT2R  = hLLDParameter[index].pT2R;

        fluxOuterLeft[index]   = getOneFlux(rhoL, uL, vL, wL, bXL, bYL, bZL, eL, pTL);
        fluxMiddleLeft[index]  = getOneFlux(rho1L, u1L, v1L, w1L, bXL, bY1L, bZ1L, e1L, pT1L);
        fluxInnerLeft[index]   = getOneFlux(rho2L, u2, v2, w2, bXL, bY2, bZ2, e2L, pT2L);
        fluxOuterRight[index]  = getOneFlux(rhoR, uR, vR, wR, bXR, bYR, bZR, eR, pTR);
        fluxMiddleRight[index] = getOneFlux(rho1R, u1R, v1R, w1R, bXR, bY1R, bZ1R, e1R, pT1R);
        fluxInnerRight[index]  = getOneFlux(rho2R, u2, v2, w2, bXR, bY2, bZ2, e2R, pT2R);

    }
}

void HLLD::setFlux()
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    setFlux_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQLeft.data()), 
        thrust::raw_pointer_cast(dQRight.data()), 
        thrust::raw_pointer_cast(hLLDParameter.data()), 
        thrust::raw_pointer_cast(fluxOuterLeft.data()), 
        thrust::raw_pointer_cast(fluxMiddleLeft.data()), 
        thrust::raw_pointer_cast(fluxInnerLeft.data()), 
        thrust::raw_pointer_cast(fluxOuterRight.data()), 
        thrust::raw_pointer_cast(fluxMiddleRight.data()), 
        thrust::raw_pointer_cast(fluxInnerRight.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY
    ); 
    cudaDeviceSynchronize();
}



thrust::device_vector<Flux>& HLLD::getFlux()
{
    return flux;
}
