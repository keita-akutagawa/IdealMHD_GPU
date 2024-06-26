#include <thrust/host_vector.h>
#include "../minmod.hpp"


int main()
{
    initializeDeviceConstants();

    thrust::host_vector<int> hVec1(3);
    hVec1[0] = 1;
    hVec1[1] = 10;
    hVec1[2] = -2;
    thrust::host_vector<int> hVec2(3);
    hVec2[0] = 10;
    hVec2[1] = -5;
    hVec2[2] = -10;

    thrust::device_vector<int> dVec1 = hVec1;
    thrust::device_vector<int> dVec2 = hVec2;
    thrust::device_vector<int> dVecResults(3, 0);

    thrust::transform(dVec1.begin(), dVec1.end(), dVec2.begin(), dVecResults.begin(), MinMod());


    thrust::host_vector<int> hVecResults(3);
    thrust::copy(dVecResults.begin(), dVecResults.end(), hVecResults.begin());

    for (int i = 0; i < hVecResults.size(); i++)
    {
        printf("%d\n", hVecResults[i]);
    }

    return 0;
}


