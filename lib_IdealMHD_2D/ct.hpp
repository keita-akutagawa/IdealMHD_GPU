#include <vector>
#include "flux_solver.hpp"


class CT
{
private:
    std::vector<std::vector<double>> EzVector;
    Flux2D oldFlux2D;

public:
    CT() : EzVector(nx, std::vector<double>(ny, 0.0)) {}

    void setOldFlux2D( 
        const Flux2D& flux2D
    );
    
    void divBClean( 
        const std::vector<std::vector<double>>& bxOld,
        const std::vector<std::vector<double>>& byOld, 
        std::vector<std::vector<std::vector<double>>>& U
    );
};

