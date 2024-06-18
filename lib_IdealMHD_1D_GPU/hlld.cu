#include "const.hpp"
#include "hlld.hpp"
#include <thrust/transform.h>
#include <thrust/tuple.h>


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
        HLLDParameter, HLLDParameter, 
        Flux, Flux, Flux, 
        Flux, Flux, Flux
        >& tupleForFlux) const {
            HLLDParameter hLLDLeftParameter = thrust::get<0>(tupleForFlux);
            HLLDParameter hLLDRightParameter = thrust::get<1>(tupleForFlux);
            Flux fluxOuterLeft = thrust::get<2>(tupleForFlux);
            Flux fluxMiddleLeft = thrust::get<3>(tupleForFlux);
            Flux fluxInnerLeft = thrust::get<4>(tupleForFlux);
            Flux fluxOuterRight = thrust::get<5>(tupleForFlux);
            Flux fluxMiddleRight = thrust::get<6>(tupleForFlux);
            Flux fluxInnerRight = thrust::get<7>(tupleForFlux);

            Flux flux;

            double SL = hLLDLeftParameter.S;
            double S1L = hLLDLeftParameter.S1;
            double SM = hLLDLeftParameter.SM; //hLLDRightParametersでもOK
            double SR = hLLDRightParameter.S;
            double S1R = hLLDRightParameter.S1;

            auto calculateFluxComponent = [&](
                double outerLeft, double middleLeft, double innerLeft,
                double outerRight, double middleRight, double innerRight
            ) {
                return outerLeft * (SL > 0.0)
                     + middleLeft * ((SL <= 0.0) && (0.0 < S1L))
                     + innerLeft * ((S1L <= 0.0) && (0.0 < SM))
                     + outerRight * (SR <= 0.0)
                     + middleRight * ((S1R <= 0.0) && (0.0 < SR))
                     + innerRight * ((SM <= 0.0) && (0.0 < S1R));
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


void HLLD::calculateFlux(
    const thrust::device_vector<ConservationParameter>& U
)
{
    setComponents(U);
    calculateHLLDParametersForOuterFan();
    calculateHLLDParametersForMiddleFan();
    calculateHLLDParametersForInnerFan();

    setFlux(outerLeftFanParameter, fluxOuterLeft);
    setFlux(outerRightFanParameter, fluxOuterRight);
    setFlux(middleLeftFanParameter, fluxMiddleLeft);
    setFlux(middleRightFanParameter, fluxMiddleRight);
    setFlux(innerLeftFanParameter, fluxInnerLeft);
    setFlux(innerRightFanParameter, fluxInnerRight);

    auto tupleForFlux = thrust::make_tuple(
        hLLDLeftParameter.begin(), hLLDRightParameter.begin(), 
        fluxOuterLeft.begin(), fluxMiddleLeft.begin(), fluxInnerLeft.begin(), 
        fluxOuterRight.begin(), fluxMiddleRight.begin(), fluxInnerRight.begin()
    );
    auto tupleForFluxIterator = thrust::make_zip_iterator(tupleForFlux);

    thrust::transform(
        tupleForFluxIterator, 
        tupleForFluxIterator + nx, 
        flux.begin(), 
        CalculateFluxFunctor()
    );
}


void HLLD::setComponents(
    const thrust::device_vector<ConservationParameter>& U
)
{
    calculateHalfQ.setPhysicalParameters(U);
    calculateHalfQ.calculateLeftQ();
    calculateHalfQ.calculateRightQ();

    dQCenter = calculateHalfQ.getCenterQ();
    dQLeft = calculateHalfQ.getLeftQ();
    dQRight = calculateHalfQ.getRightQ();
}


void HLLD::calculateHLLDParametersForOuterFan()
{
    setFanParametersFromComponents(
        dQLeft, outerLeftFanParameter
    );
    setFanParametersFromComponents(
        dQRight, outerRightFanParameter
    );
}


void HLLD::calculateHLLDParametersForMiddleFan()
{
    calculateHLLDSubParametersForMiddleFan(
        dQLeft, 
        outerLeftFanParameter, 
        hLLDLeftParameter
    );
    calculateHLLDSubParametersForMiddleFan(
        dQRight, 
        outerRightFanParameter, 
        hLLDRightParameter
    );

    double SL, SR, SM, pT1, pTL, pTR;
    double pT1L, pT1R;
    double rhoL, rhoR, uL, uR, cfL, cfR;
    for (int i = 0; i < nx; i++) {
        rhoL = outerLeftFanParameters.rho[i];
        rhoR = outerRightFanParameters.rho[i];
        uL = outerLeftFanParameters.u[i];
        uR = outerRightFanParameters.u[i];
        pTL = hLLDLeftParameters.pT[i]; //outerLeftFanParametersでもOK
        pTR = hLLDRightParameters.pT[i];
        cfL = hLLDLeftParameters.cf[i];
        cfR = hLLDRightParameters.cf[i];

        SL = std::min(uL, uR) - std::max(cfL, cfR);
        SR = std::max(uL, uR) + std::max(cfL, cfR);
        SL = std::min(SL, 0.0);
        SR = std::max(SR, 0.0);

        SM = ((SR - uR) * rhoR * uR - (SL - uL) * rhoL * uL - pTR + pTL)
           / ((SR - uR) * rhoR - (SL - uL) * rhoL + EPS);
        pT1 = ((SR - uR) * rhoR * pTL - (SL - uL) * rhoL * pTR
            + rhoL * rhoR * (SR - uR) * (SL - uL) * (uR - uL))
            / ((SR - uR) * rhoR - (SL - uL) * rhoL + EPS);
        pT1L = pT1;
        pT1R = pT1;

        hLLDLeftParameters.S[i] = SL;
        hLLDRightParameters.S[i] = SR;
        hLLDLeftParameters.SM[i] = SM;
        hLLDRightParameters.SM[i] = SM; //やらなくてもOK
        hLLDLeftParameters.pT1[i] = pT1L;
        hLLDRightParameters.pT1[i] = pT1R;
    }

    calculateHLLDParameters1(
        outerLeftFanParameter, 
        hLLDLeftParameter, 
        middleLeftFanParameter
    );
    calculateHLLDParameters1(
        outerRightFanParameter, 
        hLLDRightParameter, 
        middleRightFanParameter
    );
}


void HLLD::calculateHLLDParametersForInnerFan()
{
    double S1L, S1R, SM, rho1L, rho1R, bx1;
    for (int i = 0; i < nx; i++) {
        SM = hLLDLeftParameters.SM[i]; //RightでもOK
        rho1L = middleLeftFanParameters.rho[i];
        rho1R = middleRightFanParameters.rho[i];
        bx1 = middleLeftFanParameters.bx[i]; //RightでもOK

        S1L = SM - sqrt(bx1 * bx1 / rho1L);
        S1R = SM + sqrt(bx1 * bx1 / rho1R);

        hLLDLeftParameters.S1[i] = S1L;
        hLLDRightParameters.S1[i] = S1R;
    }

    calculateHLLDParameters2();

}


struct setFanParameterFromComponentsFunctor {

    __device__
    FanParameter operator()(const BasicParameter& dQ) const {
        FanParameter fanParameter;
        double rho, u, v, w, bX, bY, bZ, e, p, pT;

        rho = dQ.rho;
        u = dQ.u;
        v = dQ.v;
        w = dQ.w;
        bX = dQ.bX;
        bY = dQ.bY;
        bZ = dQ.bZ;
        p = dQ.p;
        e = p / (gamma_mhd - 1.0)
          + 0.5 * rho * (u * u + v * v + w * w)
          + 0.5 * (bX * bX + bY * bY + bZ * bZ); 
        pT = p + 0.5 * (bX * bX + bY * bY + bZ * bZ);

        fanParameter.rho = rho;
        fanParameter.u = u;
        fanParameter.v = v;
        fanParameter.w = w;
        fanParameter.bX = bX;
        fanParameter.bY = bY;
        fanParameter.bZ = bZ;
        fanParameter.e = e;
        fanParameter.pT = pT;

        return fanParameter;
    }
};

void HLLD::setFanParametersFromComponents(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<FanParameter>& fanParameter
)
{
    thrust::transform(
        dQ.begin(), 
        dQ.end(), 
        fanParameter.begin(), 
        setFanParameterFromComponentsFunctor()
    );
}


struct calculateHLLDSubParametersForMiddleFanFunctor {

    __device__
    HLLDParameter operator()(const BasicParameter& dQ, const FanParameter& outerFanParameter) const {
        double rho, bX, bY, bZ, e, pT, p;
        double cs, ca, va, cf;
        HLLDParameter hLLDParameter;

        rho = outerFanParameter.rho;
        bX = outerFanParameter.bX;
        bY = outerFanParameter.bY;
        bZ = outerFanParameter.bZ;
        e = outerFanParameter.e;
        pT = outerFanParameter.pT;

        p = dQ.p;

        cs = sqrt(device_gamma_mhd * p / rho);
        ca = sqrt((bX * bX + bY * bY + bZ * bZ) / rho);
        va = sqrt(bX * bX / rho);
        cf = sqrt(0.5 * (cs * cs + ca * ca
           + sqrt((cs * cs + ca * ca) * (cs * cs + ca * ca)
           - 4.0 * cs * cs * va * va)));

        hLLDParameter.pT = pT;
        hLLDParameter.e = e;
        hLLDParameter.cs = cs;
        hLLDParameter.ca = ca;
        hLLDParameter.va = va;
        hLLDParameter.cf = cf;
    
        return hLLDParameter;
    }
};

void HLLD::calculateHLLDSubParametersForMiddleFan(
    const thrust::device_vector<BasicParameter>& dQ, 
    const thrust::device_vector<FanParameter>& outerFanParameter, 
    thrust::device_vector<HLLDParameter>& hLLDParameter
)
{
    thrust::transform(
        dQ.begin(), 
        dQ.end(), 
        outerFanParameter.begin(), 
        hLLDParameter.begin(), 
        calculateHLLDSubParametersForMiddleFanFunctor()
    );
}


struct calculateHLLDParameters1Functor {

    __device__
    FanParameter operator()(const FanParameter& outerFanParameter, const HLLDParameter hLLDParameter) const {
        double rho, u, v, w, bX, bY, bZ, e, pT, pT1, S, SM;
        double rho1, u1, v1, w1, bX1, bY1, bZ1, e1;
        FanParameter middleFanParameter;

        rho = outerFanParameter.rho;
        u = outerFanParameter.u;
        v = outerFanParameter.v;
        w = outerFanParameter.w;
        bX = outerFanParameter.bX;
        bY = outerFanParameter.bY;
        bZ = outerFanParameter.bZ;

        e = hLLDParameter.e;
        pT = hLLDParameter.pT;
        pT1 = hLLDParameter.pT1;
        S = hLLDParameter.S;
        SM = hLLDParameter.SM;

        rho1 = rho * (S - u) / (S - SM + EPS);
        u1 = SM;
        v1 = v - bX * bY * (SM - u) / (rho * (S - u) * (S - SM) - bX * bX + EPS);
        w1 = w - bX * bZ * (SM - u) / (rho * (S - u) * (S - SM) - bX * bX + EPS);
        bX1 = bX;
        bY1 = bY * (rho * (S - u) * (S - u) - bX * bX)
            / (rho * (S - u) * (S - SM) - bX * bX + EPS);
        bZ1 = bZ * (rho * (S - u) * (S - u) - bX * bX)
            / (rho * (S - u) * (S - SM) - bX * bX + EPS);
        e1 = ((S - u) * e - pT * u + pT1 * SM
           + bX * ((u * bX + v * bY + w * bZ) - (u1 * bX1 + v1 * bY1 + w1 * bZ1)))
           / (S - SM + EPS);
        
        middleFanParameter.rho = rho1;
        middleFanParameter.u = u1;
        middleFanParameter.v = v1;
        middleFanParameter.w = w1;
        middleFanParameter.bX = bX1;
        middleFanParameter.bY = bY1;
        middleFanParameter.bZ = bZ1;
        middleFanParameter.e = e1;
        middleFanParameter.pT = pT1;

        return middleFanParameter;
    }
};

void HLLD::calculateHLLDParameters1(
    const thrust::device_vector<FanParameter>& outerFanParameter, 
    const thrust::device_vector<HLLDParameter>& hLLDParameter, 
    thrust::device_vector<FanParameter>& middleFanParameter
)
{
    thrust::transform(
        outerFanParameter.begin(), 
        outerFanParameter.end(), 
        hLLDParameter.begin(), 
        middleFanParameter.begin(), 
        calculateHLLDParameters1Functor()
    );
}


struct calculateHLLDParameters2Functor {

    __device__
    thrust::tuple<FanParameter, FanParameter> operator()(
        const FanParameter& middleLeftFanParameter, 
        const FanParameter& middleRightFanParameter
    ) const {
        double rho1L, u1L, v1L, w1L, bX1L, bY1L, bZ1L, e1L, pT1L;
        double rho1R, u1R, v1R, w1R, bX1R, bY1R, bZ1R, e1R, pT1R;
        double rho2L, u2L, v2L, w2L, bX2L, bY2L, bZ2L, e2L, pT2L;
        double rho2R, u2R, v2R, w2R, bX2R, bY2R, bZ2R, e2R, pT2R;
        FanParameter innerLeftFanParameter, innerRightFanParameter;
        Sign sign;

        rho1L = middleLeftFanParameter.rho;
        u1L = middleLeftFanParameter.u;
        v1L = middleLeftFanParameter.v;
        w1L = middleLeftFanParameter.w;
        bX1L = middleLeftFanParameter.bX;
        bY1L = middleLeftFanParameter.bY;
        bZ1L = middleLeftFanParameter.bZ;
        e1L = middleLeftFanParameter.e;
        pT1L = middleLeftFanParameter.pT;
        
        rho1R = middleRightFanParameter.rho;
        u1R = middleRightFanParameter.u;
        v1R = middleRightFanParameter.v;
        w1R = middleRightFanParameter.w;
        bX1R = middleRightFanParameter.bX;
        bY1R = middleRightFanParameter.bY;
        bZ1R = middleRightFanParameter.bZ;
        e1R = middleRightFanParameter.e;
        pT1R = middleRightFanParameter.pT;


        rho2L = rho1L;
        rho2R = rho1R;
        u2L = u1L;
        u2R = u1R;
        v2L = (sqrt(rho1L) * v1L + sqrt(rho1R) * v1R + (bY1R - bY1L) * sign(bX1L))
            / (sqrt(rho1L) + sqrt(rho1R) + EPS);
        v2R = v2L;
        w2L = (sqrt(rho1L) * w1L + sqrt(rho1R) * w1R + (bZ1R - bZ1L) * sign(bX1L))
            / (sqrt(rho1L) + sqrt(rho1R) + EPS);
        w2R = w2L;
        bX2L = bX1L;
        bX2R = bX1R;
        bY2L = (sqrt(rho1L) * bY1R + sqrt(rho1R) * bY1L + sqrt(rho1L * rho1R) * (v1R - v1L) * sign(bX1L))
             / (sqrt(rho1L) + sqrt(rho1R) + EPS);
        bY2R = bY2L;
        bZ2L = (sqrt(rho1L) * bZ1R + sqrt(rho1R) * bZ1L + sqrt(rho1L * rho1R) * (w1R - w1L) * sign(bX1L))
             / (sqrt(rho1L) + sqrt(rho1R) + EPS);
        bZ2R = bZ2L;
        e2L = e1L - sqrt(rho1L)
            * ((u1L * bX1L + v1L * bY1L + w1L * bZ1L) - (u2L * bX2L + v2L * bY2L + w2L * bZ2L))
            * sign(bX2L);
        e2R = e1R + sqrt(rho1R)
            * ((u1R * bX1R + v1R * bY1R + w1R * bZ1R) - (u2R * bX2R + v2R * bY2R + w2R * bZ2R))
            * sign(bX2R);
        pT2L = pT1L;
        pT2R = pT1R;

        innerLeftFanParameter.rho = rho2L;
        innerLeftFanParameter.u = u2L;
        innerLeftFanParameter.v = v2L;
        innerLeftFanParameter.w = w2L;
        innerLeftFanParameter.bX = bX2L;
        innerLeftFanParameter.bY = bY2L;
        innerLeftFanParameter.bZ = bZ2L;
        innerLeftFanParameter.e = e2L;
        innerLeftFanParameter.pT = pT2L;

        innerRightFanParameter.rho = rho2R;
        innerRightFanParameter.u = u2R;
        innerRightFanParameter.v = v2R;
        innerRightFanParameter.w = w2R;
        innerRightFanParameter.bX = bX2R;
        innerRightFanParameter.bY = bY2R;
        innerRightFanParameter.bZ = bZ2R;
        innerRightFanParameter.e = e2R;
        innerRightFanParameter.pT = pT2R;

        return thrust::make_tuple(innerLeftFanParameter, innerRightFanParameter);
    }
};

void HLLD::calculateHLLDParameters2()
{
    thrust::transform(
        middleLeftFanParameter.begin(), 
        middleLeftFanParameter.end(), 
        middleRightFanParameter.begin(), 
        thrust::make_zip_iterator(thrust::make_tuple(innerLeftFanParameter.begin(), innerRightFanParameter.begin())),
        calculateHLLDParameters2Functor()
    );
}



struct setFluxFunctor {

    __device__
    Flux operator()(const FanParameter& fanParameter) const {
        double rho, u, v, w, bX, bY, bZ, e, pT;
        Flux flux;
    
        rho = fanParameter.rho;
        u = fanParameter.u;
        v = fanParameter.v;
        w = fanParameter.w;
        bX = fanParameter.bX;
        bY = fanParameter.bY;
        bZ = fanParameter.bZ;
        e = fanParameter.e;
        pT = fanParameter.pT;

        flux.f0 = rho * u;
        flux.f1 = rho * u * u + pT - bX * bX;
        flux.f2 = rho * u * v - bX * bY;
        flux.f3 = rho * u * w - bX * bZ;
        flux.f4 = 0.0;
        flux.f5 = u * bY - v * bX;
        flux.f6 = u * bZ - w * bX;
        flux.f7 = (e + pT) * u - bX * (bX * u + bY * v + bZ * w);

        return flux;
    }
};

void HLLD::setFlux(
    const thrust::device_vector<FanParameter>& fanParameter, 
    thrust::device_vector<Flux>& flux
)
{
    thrust::transform(
        fanParameter.begin(), 
        fanParameter.end(), 
        flux.begin(), 
        setFluxFunctor()
    );
}


//getter

thrust::device_vector<Flux> HLLD::getFlux()
{
    return flux;
}
