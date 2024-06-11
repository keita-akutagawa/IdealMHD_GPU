#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "const.hpp"
#include "hlld.hpp"


inline double HLLD::sign(double x)
{
    return (x > 0.0) - (x < 0.0);
}


FanParameters::FanParameters()
{
    rho = std::vector<double>(nx, 0.0);
    u = std::vector<double>(nx, 0.0);
    v = std::vector<double>(nx, 0.0);
    w = std::vector<double>(nx, 0.0);
    bx = std::vector<double>(nx, 0.0);
    by = std::vector<double>(nx, 0.0);
    bz = std::vector<double>(nx, 0.0);
    e = std::vector<double>(nx, 0.0);
    pT = std::vector<double>(nx, 0.0);
}


HLLDParameters::HLLDParameters()
{
    pT = std::vector<double>(nx, 0.0);
    pT1 = std::vector<double>(nx, 0.0);
    pT2 = std::vector<double>(nx, 0.0);
    e = std::vector<double>(nx, 0.0);
    cs = std::vector<double>(nx, 0.0);
    ca = std::vector<double>(nx, 0.0);
    va = std::vector<double>(nx, 0.0);
    cf = std::vector<double>(nx, 0.0);
    S = std::vector<double>(nx, 0.0);
    S1 = std::vector<double>(nx, 0.0);
    SM = std::vector<double>(nx, 0.0);
}


Flux::Flux()
{
    flux = std::vector(8, std::vector<double>(nx, 0.0));
}


void HLLD::calculateFlux(
    const std::vector<std::vector<double>> U
)
{
    setComponents(U);
    calculateHLLDParametersForOuterFan();
    calculateHLLDParametersForMiddleFan();
    calculateHLLDParametersForInnerFan();

    setFlux(outerLeftFanParameters, fluxOuterLeft);
    setFlux(outerRightFanParameters, fluxOuterRight);
    setFlux(middleLeftFanParameters, fluxMiddleLeft);
    setFlux(middleRightFanParameters, fluxMiddleRight);
    setFlux(innerLeftFanParameters, fluxInnerLeft);
    setFlux(innerRightFanParameters, fluxInnerRight);

    double SL, SR, S1L, S1R, SM;
    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            SL = hLLDLeftParameters.S[i];
            S1L = hLLDLeftParameters.S1[i];
            SM = hLLDLeftParameters.SM[i]; //hLLDRightParametersでもOK
            SR = hLLDRightParameters.S[i];
            S1R = hLLDRightParameters.S1[i];

            flux.flux[comp][i] = fluxOuterLeft.flux[comp][i] * (SL > 0.0)
                               + fluxMiddleLeft.flux[comp][i] * ((SL <= 0.0) && (0.0 < S1L))
                               + fluxInnerLeft.flux[comp][i] * ((S1L <= 0.0) && (0.0 < SM))
                               + fluxOuterRight.flux[comp][i] * (SR <= 0.0)
                               + fluxMiddleRight.flux[comp][i] * ((S1R <= 0.0) && (0.0 < SR))
                               + fluxInnerRight.flux[comp][i] * ((SM <= 0.0) && (0.0 < S1R));
        }
    }
}


void HLLD::setComponents(
    const std::vector<std::vector<double>> U
)
{
    calculateHalfComponents.setPhysicalParameters(U);
    calculateHalfComponents.calculateLeftComponents();
    calculateHalfComponents.calculateRightComponents();

    componentsCenter = calculateHalfComponents.getCenterComponents();
    componentsLeft = calculateHalfComponents.getLeftComponents();
    componentsRight = calculateHalfComponents.getRightComponents();
}


void HLLD::calculateHLLDParametersForOuterFan()
{
    setFanParametersFromComponents(
        componentsLeft, outerLeftFanParameters
    );
    setFanParametersFromComponents(
        componentsRight, outerRightFanParameters
    );
}


void HLLD::calculateHLLDParametersForMiddleFan()
{
    calculateHLLDSubParametersForMiddleFan(
        componentsLeft, 
        outerLeftFanParameters, 
        hLLDLeftParameters
    );
    calculateHLLDSubParametersForMiddleFan(
        componentsRight, 
        outerRightFanParameters, 
        hLLDRightParameters
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
        outerLeftFanParameters, 
        hLLDLeftParameters, 
        middleLeftFanParameters
    );
    calculateHLLDParameters1(
        outerRightFanParameters, 
        hLLDRightParameters, 
        middleRightFanParameters
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


void HLLD::setFanParametersFromComponents(
    const Components components, 
    FanParameters& fanParameters
)
{
    double rho, u, v, w, bx, by, bz, p, e, pT;
    for (int i = 0; i < nx; i++) {
        rho = components.rho[i];
        u = components.u[i];
        v = components.v[i];
        w = components.w[i];
        bx = components.bx[i];
        by = components.by[i];
        bz = components.bz[i];
        p = components.p[i];
        e = p / (gamma_mhd - 1.0)
          + 0.5 * rho * (u * u + v * v + w * w)
          + 0.5 * (bx * bx + by * by + bz * bz); 
        pT = p + 0.5 * (bx * bx + by * by + bz * bz);
        
        fanParameters.rho[i] = rho;
        fanParameters.u[i] = u;
        fanParameters.v[i] = v;
        fanParameters.w[i] = w;
        fanParameters.bx[i] = bx;
        fanParameters.by[i] = by;
        fanParameters.bz[i] = bz;
        fanParameters.e[i] = e;
        fanParameters.pT[i] = pT;
    }
}


void HLLD::calculateHLLDSubParametersForMiddleFan(
    const Components components,
    const FanParameters outerFanParameters, 
    HLLDParameters& hLLDParameters
)
{
    double rho, bx, by, bz, e, pT, p;
    double cs, ca, va, cf;
    for (int i = 0; i < nx; i++) {
        rho = outerFanParameters.rho[i];
        bx = outerFanParameters.bx[i];
        by = outerFanParameters.by[i];
        bz = outerFanParameters.bz[i];
        e = outerFanParameters.e[i];
        pT = outerFanParameters.pT[i];

        p = components.p[i];

        cs = sqrt(gamma_mhd * p / rho);
        ca = sqrt((bx * bx + by * by + bz * bz) / rho);
        va = sqrt(bx * bx / rho);
        cf = sqrt(0.5 * (cs * cs + ca * ca
           + sqrt((cs * cs + ca * ca) * (cs * cs + ca * ca)
           - 4.0 * cs * cs * va * va)));

        hLLDParameters.pT[i] = pT;
        hLLDParameters.e[i] = e;
        hLLDParameters.cs[i] = cs;
        hLLDParameters.ca[i] = ca;
        hLLDParameters.va[i] = va;
        hLLDParameters.cf[i] = cf;
    }
}


void HLLD::calculateHLLDParameters1(
    const FanParameters outerFanParameters, 
    const HLLDParameters hlldParameters, 
    FanParameters& middleFanParameters
)
{
    double rho, u, v, w, bx, by, bz, e, pT, pT1, S, SM;
    double rho1, u1, v1, w1, bx1, by1, bz1, e1;
    for (int i = 0; i < nx; i++) {
        rho = outerFanParameters.rho[i];
        u = outerFanParameters.u[i];
        v = outerFanParameters.v[i];
        w = outerFanParameters.w[i];
        bx = outerFanParameters.bx[i];
        by = outerFanParameters.by[i];
        bz = outerFanParameters.bz[i];

        e = hlldParameters.e[i];
        pT = hlldParameters.pT[i];
        pT1 = hlldParameters.pT1[i];
        S = hlldParameters.S[i];
        SM = hlldParameters.SM[i];

        rho1 = rho * (S - u) / (S - SM + EPS);
        u1 = SM;
        v1 = v - bx * by * (SM - u) / (rho * (S - u) * (S - SM) - bx * bx + EPS);
        w1 = w - bx * bz * (SM - u) / (rho * (S - u) * (S - SM) - bx * bx + EPS);
        bx1 = bx;
        by1 = by * (rho * (S - u) * (S - u) - bx * bx)
            / (rho * (S - u) * (S - SM) - bx * bx + EPS);
        bz1 = bz * (rho * (S - u) * (S - u) - bx * bx)
            / (rho * (S - u) * (S - SM) - bx * bx + EPS);
        e1 = ((S - u) * e - pT * u + pT1 * SM
           + bx * ((u * bx + v * by + w * bz) - (u1 * bx1 + v1 * by1 + w1 * bz1)))
           / (S - SM + EPS);
        
        middleFanParameters.rho[i] = rho1;
        middleFanParameters.u[i] = u1;
        middleFanParameters.v[i] = v1;
        middleFanParameters.w[i] = w1;
        middleFanParameters.bx[i] = bx1;
        middleFanParameters.by[i] = by1;
        middleFanParameters.bz[i] = bz1;
        middleFanParameters.e[i] = e1;
        middleFanParameters.pT[i] = pT1;
    }
}


void HLLD::calculateHLLDParameters2()
{
    double rho1L, u1L, v1L, w1L, bx1L, by1L, bz1L, e1L, pT1L;
    double rho1R, u1R, v1R, w1R, bx1R, by1R, bz1R, e1R, pT1R;
    double rho2L, u2L, v2L, w2L, bx2L, by2L, bz2L, e2L, pT2L;
    double rho2R, u2R, v2R, w2R, bx2R, by2R, bz2R, e2R, pT2R;

    for (int i = 0; i < nx; i++) {
        rho1L = middleLeftFanParameters.rho[i];
        u1L = middleLeftFanParameters.u[i];
        v1L = middleLeftFanParameters.v[i];
        w1L = middleLeftFanParameters.w[i];
        bx1L = middleLeftFanParameters.bx[i];
        by1L = middleLeftFanParameters.by[i];
        bz1L = middleLeftFanParameters.bz[i];
        e1L = middleLeftFanParameters.e[i];
        pT1L = middleLeftFanParameters.pT[i];
        
        rho1R = middleRightFanParameters.rho[i];
        u1R = middleRightFanParameters.u[i];
        v1R = middleRightFanParameters.v[i];
        w1R = middleRightFanParameters.w[i];
        bx1R = middleRightFanParameters.bx[i];
        by1R = middleRightFanParameters.by[i];
        bz1R = middleRightFanParameters.bz[i];
        e1R = middleRightFanParameters.e[i];
        pT1R = middleRightFanParameters.pT[i];


        rho2L = rho1L;
        rho2R = rho1R;
        u2L = u1L;
        u2R = u1R;
        v2L = (sqrt(rho1L) * v1L + sqrt(rho1R) * v1R + (by1R - by1L) * sign(bx1L))
            / (sqrt(rho1L) + sqrt(rho1R) + EPS);
        v2R = v2L;
        w2L = (sqrt(rho1L) * w1L + sqrt(rho1R) * w1R + (bz1R - bz1L) * sign(bx1L))
            / (sqrt(rho1L) + sqrt(rho1R) + EPS);
        w2R = w2L;
        bx2L = bx1L;
        bx2R = bx1R;
        by2L = (sqrt(rho1L) * by1R + sqrt(rho1R) * by1L + sqrt(rho1L * rho1R) * (v1R - v1L) * sign(bx1L))
             / (sqrt(rho1L) + sqrt(rho1R) + EPS);
        by2R = by2L;
        bz2L = (sqrt(rho1L) * bz1R + sqrt(rho1R) * bz1L + sqrt(rho1L * rho1R) * (w1R - w1L) * sign(bx1L))
             / (sqrt(rho1L) + sqrt(rho1R) + EPS);
        bz2R = bz2L;
        e2L = e1L - sqrt(rho1L)
            * ((u1L * bx1L + v1L * by1L + w1L * bz1L) - (u2L * bx2L + v2L * by2L + w2L * bz2L))
            * sign(bx2L);
        e2R = e1R + sqrt(rho1R)
            * ((u1R * bx1R + v1R * by1R + w1R * bz1R) - (u2R * bx2R + v2R * by2R + w2R * bz2R))
            * sign(bx2R);
        pT2L = pT1L;
        pT2R = pT1R;

        innerLeftFanParameters.rho[i] = rho2L;
        innerLeftFanParameters.u[i] = u2L;
        innerLeftFanParameters.v[i] = v2L;
        innerLeftFanParameters.w[i] = w2L;
        innerLeftFanParameters.bx[i] = bx2L;
        innerLeftFanParameters.by[i] = by2L;
        innerLeftFanParameters.bz[i] = bz2L;
        innerLeftFanParameters.e[i] = e2L;
        innerLeftFanParameters.pT[i] = pT2L;

        innerRightFanParameters.rho[i] = rho2R;
        innerRightFanParameters.u[i] = u2R;
        innerRightFanParameters.v[i] = v2R;
        innerRightFanParameters.w[i] = w2R;
        innerRightFanParameters.bx[i] = bx2R;
        innerRightFanParameters.by[i] = by2R;
        innerRightFanParameters.bz[i] = bz2R;
        innerRightFanParameters.e[i] = e2R;
        innerRightFanParameters.pT[i] = pT2R;

        hLLDLeftParameters.pT2[i] = pT2L;
        hLLDRightParameters.pT2[i] = pT2R;
    }
}


void HLLD::setFlux(
    const FanParameters fanParameters, 
    Flux& flux
)
{
    double rho, u, v, w, bx, by, bz, e;
    double pT;
    for (int i = 0; i < nx; i++) {
        rho = fanParameters.rho[i];
        u = fanParameters.u[i];
        v = fanParameters.v[i];
        w = fanParameters.w[i];
        bx = fanParameters.bx[i];
        by = fanParameters.by[i];
        bz = fanParameters.bz[i];
        e = fanParameters.e[i];
        pT = fanParameters.pT[i];

        flux.flux[0][i] = rho * u;
        flux.flux[1][i] = rho * u * u + pT - bx * bx;
        flux.flux[2][i] = rho * u * v - bx * by;
        flux.flux[3][i] = rho * u * w - bx * bz;
        flux.flux[4][i] = 0.0;
        flux.flux[5][i] = u * by - v * bx;
        flux.flux[6][i] = u * bz - w * bx;
        flux.flux[7][i] = (e + pT) * u - bx * (bx * u + by * v + bz * w);
    }
}

// getter

Components HLLD::getLeftComponents()
{
    return componentsLeft;
}

Components HLLD::getRightComponents()
{
    return componentsRight;
}

FanParameters HLLD::getOuterLeftFanParameters()
{
    return outerLeftFanParameters;
}

FanParameters HLLD::getOuterRightFanParameters()
{
    return outerRightFanParameters;
}

FanParameters HLLD::getMiddleLeftFanParameters()
{
    return middleLeftFanParameters;
}

FanParameters HLLD::getMiddleRightFanParameters()
{
    return middleRightFanParameters;
}

FanParameters HLLD::getInnerLeftFanParameters()
{
    return innerLeftFanParameters;
}

FanParameters HLLD::getInnerRightFanParameters()
{
    return innerRightFanParameters;
}

HLLDParameters HLLD::getHLLDLeftParameters()
{
    return hLLDLeftParameters;
}

HLLDParameters HLLD::getHLLDRightParameters()
{
    return hLLDRightParameters;
}

Flux HLLD::getFlux()
{
    return flux;
}
