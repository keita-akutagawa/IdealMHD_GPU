#include <vector>
#include "const.hpp"
#include "calculate_half_components.hpp"


struct FanParameters
{
    std::vector<double> rho;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> w;
    std::vector<double> bx;
    std::vector<double> by;
    std::vector<double> bz;
    std::vector<double> e;
    std::vector<double> pT;

    FanParameters(int nSize) : 
        rho(nSize, 0.0),   
        u(nSize, 0.0), v(nSize, 0.0), w(nSize, 0.0), 
        bx(nSize, 0.0), by(nSize, 0.0), bz(nSize, 0.0), 
        e(nSize, 0.0), pT(nSize, 0.0) 
        {};
};

struct HLLDParameters
{
    std::vector<double> pT;
    std::vector<double> pT1;
    std::vector<double> pT2;
    std::vector<double> e;
    std::vector<double> cs;
    std::vector<double> ca;
    std::vector<double> va;
    std::vector<double> cf;
    std::vector<double> S;
    std::vector<double> S1;
    std::vector<double> SM;

    HLLDParameters(int nSize) : 
        pT(nSize, 0.0), pT1(nSize, 0.0), pT2(nSize, 0.0),  
        e(nSize, 0.0), cs(nSize, 0.0), ca(nSize, 0.0), 
        va(nSize, 0.0), cf(nSize, 0.0), 
        S(nSize, 0.0), S1(nSize, 0.0), SM(nSize, 0.0) 
        {};
};

struct Flux1D
{ 
    std::vector<std::vector<double>> flux;

    Flux1D(int nSize) : flux(8, std::vector<double>(nSize, 0.0)) {};
};




class HLLD
{
private:
    int nDirection;

    CalculateHalfComponents calculateHalfComponents;
    Components componentsCenter;
    Components componentsLeft;
    Components componentsRight;
    FanParameters outerLeftFanParameters;
    FanParameters outerRightFanParameters;
    FanParameters middleLeftFanParameters;
    FanParameters middleRightFanParameters;
    FanParameters innerLeftFanParameters;
    FanParameters innerRightFanParameters;
    HLLDParameters hLLDLeftParameters;
    HLLDParameters hLLDRightParameters;

    Flux1D flux;
    Flux1D fluxOuterLeft, fluxMiddleLeft, fluxInnerLeft;
    Flux1D fluxOuterRight, fluxMiddleRight, fluxInnerRight;

public:
    HLLD(int nSize) : 
        nDirection(nSize), 
        calculateHalfComponents(nSize),
        componentsCenter(nSize),
        componentsLeft(nSize),
        componentsRight(nSize),
        outerLeftFanParameters(nSize),
        outerRightFanParameters(nSize),
        middleLeftFanParameters(nSize),
        middleRightFanParameters(nSize),
        innerLeftFanParameters(nSize),
        innerRightFanParameters(nSize),
        hLLDLeftParameters(nSize), 
        hLLDRightParameters(nSize), 
        flux(nSize),
        fluxOuterLeft(nSize), fluxMiddleLeft(nSize), fluxInnerLeft(nSize), 
        fluxOuterRight(nSize), fluxMiddleRight(nSize), fluxInnerRight(nSize)
        {}

    void calculateFlux(
        const std::vector<std::vector<double>>& U
    );

    Components getLeftComponents();
    Components getRightComponents();

    FanParameters getOuterLeftFanParameters();
    FanParameters getOuterRightFanParameters();
    FanParameters getMiddleLeftFanParameters();
    FanParameters getMiddleRightFanParameters();
    FanParameters getInnerLeftFanParameters();
    FanParameters getInnerRightFanParameters();

    HLLDParameters getHLLDLeftParameters();
    HLLDParameters getHLLDRightParameters();

    Flux1D getFlux();

private:
    double sign(double x);

    void setComponents(
        const std::vector<std::vector<double>>& U
    );

    void calculateHLLDParametersForOuterFan();

    void calculateHLLDParametersForMiddleFan();

    void calculateHLLDParametersForInnerFan();

    void setFanParametersFromComponents(
        const Components& components, 
        FanParameters& fanParameters
    );

    void calculateHLLDSubParametersForMiddleFan(
        const Components& components, 
        const FanParameters& outerFanParameters, 
        HLLDParameters& hLLDParameters
    );

    void calculateHLLDParameters1(
        const FanParameters& outerFanParameters, 
        const HLLDParameters& hLLDParameters, 
        FanParameters& middleFanParameters
    );

    void calculateHLLDParameters2();

    void setFlux(
        const FanParameters& fanParameters, 
        Flux1D& flux
    );
};

