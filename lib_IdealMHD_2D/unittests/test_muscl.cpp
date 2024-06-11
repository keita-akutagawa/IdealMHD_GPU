#include <gtest/gtest.h>
#include "../muscl.hpp"
#include "../const.hpp"
#include <algorithm>
#include <vector>


TEST(MUSCLTest, CheckConstInputConstOutput)
{
    std::vector<double> q = std::vector<double>(nx, 1.0);
    std::vector<double> qLeft = std::vector<double>(nx, 0.0);
    std::vector<double> qRight = std::vector<double>(nx, 0.0);

    MUSCL muscl;

    muscl.getLeftComponent(q, qLeft);
    muscl.getRightComponent(q, qRight);

    for (int i = 0; i < nx; i++) {
        EXPECT_NEAR(qLeft[i], q[i], 1e-10);
        EXPECT_NEAR(qRight[i], q[i], 1e-10);
    }
}

TEST(MUSCLTest, CheckDeltaInputChangeOutput)
{
    std::vector<double> q = std::vector<double>(nx, 0.0);
    std::vector<double> qLeft = std::vector<double>(nx, 0.0);
    std::vector<double> qRight = std::vector<double>(nx, 0.0);
    q[int(nx / 2.0)] = 1.0;

    MUSCL muscl;

    muscl.getLeftComponent(q, qLeft);
    muscl.getRightComponent(q, qRight);

    for (int i = 0; i < nx; i++) {
        EXPECT_NEAR(qLeft[i], q[i], 1e-10);
        EXPECT_NEAR(qRight[i], q[i], 1e-10);
    }
}



