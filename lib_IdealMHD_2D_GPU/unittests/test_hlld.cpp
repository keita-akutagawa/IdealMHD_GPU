#include <gtest/gtest.h>
#include "../hlld.hpp"
#include "../const.hpp"
#include <algorithm>
#include <vector>
#include <cmath>


TEST(HLLDTest, CheckStructConstructor)
{
    HLLD hLLD;
    Components components;
    FanParameters fanParameters;
    HLLDParameters hLLDParameters;
    Flux flux;

    components = hLLD.getLeftComponents();
    fanParameters = hLLD.getOuterLeftFanParameters();
    hLLDParameters = hLLD.getHLLDLeftParameters();
    flux = hLLD.getFlux();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(components.rho[i], 0.0);
        EXPECT_EQ(fanParameters.pT[i], 0.0);
        EXPECT_EQ(hLLDParameters.ca[i], 0.0);
    }

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            EXPECT_EQ(flux.flux[comp][i], 0.0);
        }
    }
}

TEST(HLLDTest, SetComponents)
{
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 1.0));

    HLLD hLLD;
    Components components;
    FanParameters fanParameters;
    HLLDParameters hLLDParameters;
    Flux flux;

    hLLD.calculateFlux(U);

    components = hLLD.getLeftComponents();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(components.rho[i], 1.0); 
        EXPECT_EQ(components.bx[i], 1.0); 
    }
}

TEST(HLLDTest, CalculateFanParameters)
{
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 1.0));

    HLLD hLLD;
    Components components;
    FanParameters fanParameters;
    HLLDParameters hLLDParameters;
    Flux flux;

    hLLD.calculateFlux(U);
    
    fanParameters = hLLD.getOuterLeftFanParameters();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(fanParameters.rho[i], 1.0); 
        EXPECT_NE(fanParameters.pT[i], 0.0); 
    }
}

TEST(HLLDTest, CalculateHLLDParameters)
{
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 1.0));

    HLLD hLLD;
    Components components;
    FanParameters fanParameters;
    HLLDParameters hLLDParameters;
    Flux flux;

    hLLD.calculateFlux(U);
    
    hLLDParameters = hLLD.getHLLDLeftParameters();

    for (int i = 0; i < nx; i++) {
        EXPECT_NE(hLLDParameters.pT[i], 0.0); 
        EXPECT_NE(hLLDParameters.S[i], 0.0); 
    }
}

TEST(HLLDTest, CalculateFlux)
{
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 1.0));

    HLLD hLLD;
    Components components;
    FanParameters fanParameters;
    HLLDParameters hLLDParameters;
    Flux flux;

    hLLD.calculateFlux(U);
    
    flux = hLLD.getFlux();

    for (int i = 0; i < nx; i++) {
        EXPECT_NE(flux.flux[0][i], 0.0); //flux.flux[4][i]は0.0
    }
}


TEST(HLLDTest, CalculateFluxDiscontinuity)
{
    double rhoL0, uL0, vL0, wL0, bxL0, byL0, bzL0, pL0, eL0;
    double rhoR0, uR0, vR0, wR0, bxR0, byR0, bzR0, pR0, eR0;

    rhoL0 = 1.0;
    uL0 = 0.0; vL0 = 0.0; wL0 = 0.0;
    bxL0 = 0.75; byL0 = 1.0; bzL0 = 0.0;
    pL0 = 1.0;
    eL0 = pL0 / (gamma_mhd - 1.0)
        + 0.5 * rhoL0 * (uL0 * uL0 + vL0 * vL0 + wL0 * wL0)
        + 0.5 * (bxL0 * bxL0 + byL0 * byL0 + bzL0 * bzL0);

    rhoR0 = 0.125;
    uR0 = 0.0; vR0 = 0.0; wR0 = 0.0;
    bxR0 = 0.75; byR0 = -1.0; bzR0 = 0.0;
    pR0 = 0.1;
    eR0 = pR0 / (gamma_mhd - 1.0)
        + 0.5 * rhoR0 * (uR0 * uR0 + vR0 * vR0 + wR0 * wR0)
        + 0.5 * (bxR0 * bxR0 + byR0 * byR0 + bzR0 * bzR0);

    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 0.0));
    for (int i = 0; i < int(nx / 2.0); i++) {
        U[0][i] = rhoL0;
        U[1][i] = uL0;
        U[2][i] = vL0;
        U[3][i] = wL0;
        U[4][i] = bxL0;
        U[5][i] = byL0;
        U[6][i] = bzL0;
        U[7][i] = eL0;
    }
    for (int i = int(nx / 2.0); i < nx; i++) {
        U[0][i] = rhoR0;
        U[1][i] = uR0;
        U[2][i] = vR0;
        U[3][i] = wR0;
        U[4][i] = bxR0;
        U[5][i] = byR0;
        U[6][i] = bzR0;
        U[7][i] = eR0;
    }


    HLLD hLLD;
    Components components;
    FanParameters fanParameters;
    HLLDParameters hLLDParameters;
    Flux flux;

    hLLD.calculateFlux(U);
    
    flux = hLLD.getFlux();

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            EXPECT_FALSE(std::isnan(flux.flux[comp][i]));
        }
    }
    
}



