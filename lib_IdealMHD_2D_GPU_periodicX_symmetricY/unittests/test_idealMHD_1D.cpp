#include <gtest/gtest.h>
#include "../idealMHD_1D.hpp"
#include "../const.hpp"
#include <algorithm>
#include <vector>
#include <cmath>


TEST(IdealMHD1D, initializeU)
{
    std::vector<std::vector<double>> UInit(8, std::vector<double>(nx, 1.0));
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 0.0));

    IdealMHD1D idealMHD1D;

    idealMHD1D.initializeU(UInit);

    U = idealMHD1D.getU();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(U[0][i], 1.0);
    }
}

TEST(IdealMHD1D, calculateDt)
{
    double rho, u, v, w, bx, by, bz, p, e;
    rho = u = v = w = bx = by = bz = p = 1.0;
    e = p / (gamma_mhd - 1.0)
      + 0.5 * rho * (u * u + v * v + w * w)
      + 0.5 * (bx * bx + by * by + bz * bz);

    std::vector<std::vector<double>> UInit(8, std::vector<double>(nx, 0.0));
    for (int i = 0; i < nx; i++) {
        UInit[0][i] = rho;
        UInit[1][i] = u;
        UInit[2][i] = v;
        UInit[3][i] = w;
        UInit[4][i] = bx;
        UInit[5][i] = by;
        UInit[6][i] = bz;
        UInit[7][i] = e;
    }

    IdealMHD1D idealMHD1D;

    idealMHD1D.initializeU(UInit);

    idealMHD1D.calculateDt();

    EXPECT_FALSE(std::isnan(dt));
}

TEST(IdealMHD1D, oneStepRK2)
{
    double rho, u, v, w, bx, by, bz, p, e;
    rho = u = v = w = bx = by = bz = p = 1.0;
    e = p / (gamma_mhd - 1.0)
      + 0.5 * rho * (u * u + v * v + w * w)
      + 0.5 * (bx * bx + by * by + bz * bz);

    std::vector<std::vector<double>> UInit(8, std::vector<double>(nx, 0.0));
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 0.0));
    for (int i = 0; i < nx; i++) {
        UInit[0][i] = rho;
        UInit[1][i] = u;
        UInit[2][i] = v;
        UInit[3][i] = w;
        UInit[4][i] = bx;
        UInit[5][i] = by;
        UInit[6][i] = bz;
        UInit[7][i] = e;
    }

    IdealMHD1D idealMHD1D;

    idealMHD1D.initializeU(UInit);

    idealMHD1D.oneStepRK2();

    U = idealMHD1D.getU();

    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            EXPECT_FALSE(std::isnan(U[comp][i]));
        }
    }
}

TEST(IdealMHD1D, oneStepRK2Discontinuity)
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

    std::vector<std::vector<double>> UInit(8, std::vector<double>(nx, 0.0));
    for (int i = 0; i < int(nx / 2.0); i++) {
        UInit[0][i] = rhoL0;
        UInit[1][i] = uL0;
        UInit[2][i] = vL0;
        UInit[3][i] = wL0;
        UInit[4][i] = bxL0;
        UInit[5][i] = byL0;
        UInit[6][i] = bzL0;
        UInit[7][i] = eL0;
    }
    for (int i = int(nx / 2.0); i < nx; i++) {
        UInit[0][i] = rhoR0;
        UInit[1][i] = uR0;
        UInit[2][i] = vR0;
        UInit[3][i] = wR0;
        UInit[4][i] = bxR0;
        UInit[5][i] = byR0;
        UInit[6][i] = bzR0;
        UInit[7][i] = eR0;
    }


    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 0.0));
    IdealMHD1D idealMHD1D;

    idealMHD1D.initializeU(UInit);

    idealMHD1D.oneStepRK2();

    U = idealMHD1D.getU();
    
    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            EXPECT_FALSE(std::isnan(U[comp][i]));
        }
    }
}


