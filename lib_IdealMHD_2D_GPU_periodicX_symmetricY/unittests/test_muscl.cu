#include <thrust/host_vector.h>
#include "../muscl.hpp"
#include <iostream>


int main()
{
    initializeDeviceConstants();

    thrust::host_vector<BasicParameter> hQ(nx);
    for (int i = 0; i < nx; i++) {
        hQ[i].rho = i;
    }

    thrust::device_vector<BasicParameter> dQ = hQ;
    thrust::device_vector<BasicParameter> dLeft(nx);
    thrust::device_vector<BasicParameter> dRight(nx);


    MUSCL muscl;
    muscl.getLeftComponent(dQ, dLeft);
    muscl.getRightComponent(dQ, dRight);


    thrust::host_vector<BasicParameter> hLeft = dLeft;
    thrust::host_vector<BasicParameter> hRight = dRight;

    std::cout << "left part" << std::endl;
    for (int i = 0; i < nx; i++)
    {
        std::cout << hLeft[i].rho << std::endl;
    }

    std::cout << "right part" << std::endl;
    for (int i = 0; i < nx; i++)
    {
        std::cout << hRight[i].rho << std::endl;
    }

    return 0;
}





