#include <vector>
#include "const.hpp"
#include "calculate_half_components.hpp"


class CalculateHalfComponents;


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

    FanParameters();
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

    HLLDParameters();
};

struct Flux
{
    std::vector<std::vector<double>> flux;

    Flux();
};




class HLLD
{
private:
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

    Flux flux;
    Flux fluxOuterLeft, fluxMiddleLeft, fluxInnerLeft;
    Flux fluxOuterRight, fluxMiddleRight, fluxInnerRight;

public:

    void calculateFlux(
        const std::vector<std::vector<double>> U
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

    Flux getFlux();

private:
    double sign(double x);

    void setComponents(
        const std::vector<std::vector<double>> U
    );

    void calculateHLLDParametersForOuterFan();

    void calculateHLLDParametersForMiddleFan();

    void calculateHLLDParametersForInnerFan();

    void setFanParametersFromComponents(
        const Components components, 
        FanParameters& fanParameters
    );

    void calculateHLLDSubParametersForMiddleFan(
        const Components components, 
        const FanParameters outerFanParameters, 
        HLLDParameters& hLLDParameters
    );

    void calculateHLLDParameters1(
        const FanParameters outerFanParameters, 
        const HLLDParameters hLLDParameters, 
        FanParameters& middleFanParameters
    );

    void calculateHLLDParameters2();

    void setFlux(
        const FanParameters fanParameters, 
        Flux& flux
    );
};

