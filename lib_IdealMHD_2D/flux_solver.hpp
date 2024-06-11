#include "const.hpp"
#include "hlld.hpp"


struct Flux2D
{
    std::vector<std::vector<std::vector<double>>> fluxF;
    std::vector<std::vector<std::vector<double>>> fluxG;

    Flux2D() : 
        fluxF(8, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        fluxG(8, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0)))
        {}
};


class FluxSolver
{
private:
    HLLD hLLDForF, hLLDForG;
    Flux1D flux1DForF, flux1DForG;
    Flux2D flux2D;
    std::vector<std::vector<double>> tmpUForF;
    std::vector<std::vector<double>> tmpUForG;
    std::vector<std::vector<std::vector<double>>> tmpFlux;

public:
    FluxSolver() : 
        hLLDForF(nx), 
        hLLDForG(ny), 
        flux1DForF(nx), 
        flux1DForG(ny), 
        tmpUForF(8, std::vector<double>(nx, 0.0)), 
        tmpUForG(8, std::vector<double>(ny, 0.0)), 
        tmpFlux(8, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0)))
        {};

    Flux2D getFluxF(
        const std::vector<std::vector<std::vector<double>>>& U
    );

    Flux2D getFluxG(
        const std::vector<std::vector<std::vector<double>>>& U
    );

    void setFluxGToProperPosition();
};


