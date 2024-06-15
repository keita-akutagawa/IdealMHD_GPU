#include <thrust/host_vector.h>
#include "../muscl.hpp"
#include <iostream>


int main()
{
    initializeDeviceConstants();

    thrust::host_vector<double> hQ(nx, 0.0);
    for (int i = 0; i < nx; i++) {
        hQ[i] = i;
    }

    thrust::device_vector<double> dQ = hQ;
    thrust::device_vector<double> dLeft(nx, 0.0);
    thrust::device_vector<double> dRight(nx, 0.0);


    MUSCL muscl;
    muscl.getLeftComponent(dQ, dLeft);
    muscl.getRightComponent(dQ, dRight);


    thrust::host_vector<double> hLeft = dLeft;
    thrust::host_vector<double> hRight = dRight;

    std::cout << "left part" << std::endl;
    for (int i = 0; i < nx; i++)
    {
        std::cout << hLeft[i] << std::endl;
    }

    std::cout << "right part" << std::endl;
    for (int i = 0; i < nx; i++)
    {
        std::cout << hRight[i] << std::endl;
    }

    return 0;
}





