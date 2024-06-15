#include <thrust/device_vector.h>
#include "const.hpp"
#include "muscl.hpp"



struct Components
{
    thrust::device_vector<double> rho;
    thrust::device_vector<double> u;
    thrust::device_vector<double> v;
    thrust::device_vector<double> w;
    thrust::device_vector<double> bx;
    thrust::device_vector<double> by;
    thrust::device_vector<double> bz;
    thrust::device_vector<double> p;

    Components() : 
        rho(nx, 0.0),
        u(nx, 0.0),
        v(nx, 0.0),
        w(nx, 0.0),
        bx(nx, 0.0),
        by(nx, 0.0),
        bz(nx, 0.0),
        p(nx, 0.0)
        {}
};


class CalculateHalfComponents
{
private:
    MUSCL muscl;

    Components componentsCenter;
    Components componentsLeft;
    Components componentsRight;

public:

    void setPhysicalParameters(
        const thrust::device_vector<thrust::device_vector<double>>& U
    );

    void calculateLeftComponents();

    void calculateRightComponents();

    Components getCenterComponents();

    Components getLeftComponents();

    Components getRightComponents();
};

