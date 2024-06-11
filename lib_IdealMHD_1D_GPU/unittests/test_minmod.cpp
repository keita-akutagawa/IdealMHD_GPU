#include <gtest/gtest.h>
#include "../minmod.hpp"
#include <algorithm>


TEST(MinmodTest, Zero)
{
    EXPECT_NEAR(minmod(1.0, -0.1), 0.0, 1e-10);
    EXPECT_NEAR(minmod(-1.0, 0.1), 0.0, 1e-10);
}

TEST(MinmodTest, Positive)
{
    EXPECT_NEAR(minmod(1.0, 0.1), 0.1, 1e-10);
    EXPECT_NEAR(minmod(0.1, 1.0), 0.1, 1e-10);
    EXPECT_NEAR(minmod(-1.0, -0.1), -0.1, 1e-10);
    EXPECT_NEAR(minmod(-0.1, -1.0), -0.1, 1e-10);
}


