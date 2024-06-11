#include <gtest/gtest.h>
#include "../calculate_half_components.hpp"
#include "../const.hpp"
#include <algorithm>
#include <vector>


TEST(CalculateHalfComponentsTest, Constructor)
{
    CalculateHalfComponents calculateHalfComponents;

    Components componentsCenter, componentsLeft, componentsRight;
    componentsCenter = calculateHalfComponents.getCenterComponents();
    componentsLeft = calculateHalfComponents.getLeftComponents();
    componentsRight = calculateHalfComponents.getRightComponents();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(componentsCenter.rho[i], 0.0);
        EXPECT_EQ(componentsLeft.rho[i], 0.0);
        EXPECT_EQ(componentsRight.rho[i], 0.0);
    }
}


TEST(CalculateHalfComponentsTest, setPhysicalParameters)
{
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 1.0));

    CalculateHalfComponents calculateHalfComponents;

    calculateHalfComponents.setPhysicalParameters(U);

    Components componentsCenter, componentsLeft, componentsRight;
    componentsCenter = calculateHalfComponents.getCenterComponents();
    componentsLeft = calculateHalfComponents.getLeftComponents();
    componentsRight = calculateHalfComponents.getRightComponents();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(componentsCenter.rho[i], 1.0);
        EXPECT_EQ(componentsLeft.rho[i], 0.0);
        EXPECT_EQ(componentsRight.rho[i], 0.0);
    }
}


TEST(CalculateHalfComponentsTest, getLeftRightParameters)
{
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 1.0));

    CalculateHalfComponents calculateHalfComponents;

    calculateHalfComponents.setPhysicalParameters(U);
    calculateHalfComponents.calculateLeftComponents();
    calculateHalfComponents.calculateRightComponents();

    Components componentsCenter, componentsLeft, componentsRight;
    componentsCenter = calculateHalfComponents.getCenterComponents();
    componentsLeft = calculateHalfComponents.getLeftComponents();
    componentsRight = calculateHalfComponents.getRightComponents();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(componentsCenter.rho[i], 1.0);
        EXPECT_EQ(componentsLeft.rho[i], 1.0);
        EXPECT_EQ(componentsRight.rho[i], 1.0);
    }
}


TEST(CalculateHalfComponentsTest, getLeftRightParametersAgain)
{
    std::vector<std::vector<double>> U(8, std::vector<double>(nx, 1.0));

    CalculateHalfComponents calculateHalfComponents;

    calculateHalfComponents.setPhysicalParameters(U);
    calculateHalfComponents.calculateLeftComponents();
    calculateHalfComponents.calculateRightComponents();

    Components componentsCenter, componentsLeft, componentsRight;
    componentsCenter = calculateHalfComponents.getCenterComponents();
    componentsLeft = calculateHalfComponents.getLeftComponents();
    componentsRight = calculateHalfComponents.getRightComponents();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(componentsCenter.rho[i], 1.0);
        EXPECT_EQ(componentsLeft.rho[i], 1.0);
        EXPECT_EQ(componentsRight.rho[i], 1.0);
    }


    for (int comp = 0; comp < 8; comp++) {
        for (int i = 0; i < nx; i++) {
            U[comp][i] = 2.0;
        }
    }

    calculateHalfComponents.setPhysicalParameters(U);
    calculateHalfComponents.calculateLeftComponents();
    calculateHalfComponents.calculateRightComponents();

    componentsCenter = calculateHalfComponents.getCenterComponents();
    componentsLeft = calculateHalfComponents.getLeftComponents();
    componentsRight = calculateHalfComponents.getRightComponents();

    for (int i = 0; i < nx; i++) {
        EXPECT_EQ(componentsCenter.rho[i], 2.0);
        EXPECT_EQ(componentsLeft.rho[i], 2.0);
        EXPECT_EQ(componentsRight.rho[i], 2.0);
    }
}

