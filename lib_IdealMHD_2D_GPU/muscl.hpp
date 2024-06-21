#include <thrust/device_vector.h>
#include "const.hpp"
#include "minmod.hpp"
#include "basic_parameter_struct.hpp"


class MUSCL
{
private:
    int nDirection;

public:
    MUSCL(int nSize) : nDirection(nSize) {}

    void getLeftQX(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQLeft
    );

    void getLeftQY(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQLeft
    );

    void getRightQX(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQRight
    );

    void getRightQY(
        const thrust::device_vector<BasicParameter>& dQ, 
        thrust::device_vector<BasicParameter>& dQRight
    );
};

