#include <thrust/host_vector.h>
#include "../muscl.hpp"
#include <iostream>


int main()
{
    std::cout << nx << std::endl;
    
    initializeDeviceConstants();

    MUSCL muscl;

    thrust::host_vector<double> hQ(nx, 0.0);
    for (int i = 0; i < nx; i++) {
        hQ[i] = i;
    }

    thrust::device_vector<double> dQ = hQ;
    thrust::device_vector<double> dLeft(nx, 0.0);


    muscl.getLeftComponent(dQ, dLeft);


    thrust::host_vector<double> hLeft = dLeft;

    for (int i = 0; i < nx; i++)
    {
        std::cout << hLeft[i] << std::endl;
    }

    std::cout << "OK" << std::endl;
    return 0;
}





