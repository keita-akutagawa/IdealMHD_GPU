#include <vector>
#include "const.hpp"
#include "muscl.hpp"



struct Components
{ 
    std::vector<double> rho;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> w;
    std::vector<double> bx;
    std::vector<double> by;
    std::vector<double> bz;
    std::vector<double> p;

    Components(int nSize) : 
        rho(nSize, 0.0),   
        u(nSize, 0.0), v(nSize, 0.0), w(nSize, 0.0), 
        bx(nSize, 0.0), by(nSize, 0.0), bz(nSize, 0.0), 
        p(nSize, 0.0)
        {};
};


class CalculateHalfComponents
{
private:
    int nDirection; 

    MUSCL muscl;

    Components componentsCenter;
    Components componentsLeft;
    Components componentsRight;

public:
    CalculateHalfComponents(int nSize) : 
        nDirection(nSize), 
        muscl(nSize), 
        componentsCenter(nSize), 
        componentsLeft(nSize), 
        componentsRight(nSize)
        {}
    
    void setPhysicalParameters(
        const std::vector<std::vector<double>>& U
    );

    void calculateLeftComponents();

    void calculateRightComponents();

    Components getCenterComponents();

    Components getLeftComponents();

    Components getRightComponents();
};

