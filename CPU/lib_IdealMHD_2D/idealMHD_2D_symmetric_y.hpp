#include <vector>
#include <string>
#include "const.hpp"
//#include "flux_solver.hpp" // divB_cleanerのCT法でFlux2Dを使うときにincludeした
#include "boundary.hpp"
#include "ct.hpp"


class IdealMHD2DSymmetricY
{
private:
    FluxSolver fluxSolver;
    Flux2D flux2D;
    std::vector<std::vector<std::vector<double>>> U;
    std::vector<std::vector<std::vector<double>>> UBar;
    Boundary boundary;
    std::vector<std::vector<double>> bxOld;
    std::vector<std::vector<double>> byOld;
    std::vector<std::vector<double>> tmpVector;
    CT ct;

public:
    IdealMHD2DSymmetricY() :
        U(8, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        UBar(8, std::vector<std::vector<double>>(nx, std::vector<double>(ny, 0.0))), 
        bxOld(nx, std::vector<double>(ny, 0.0)), 
        byOld(nx, std::vector<double>(ny, 0.0)), 
        tmpVector(nx, std::vector<double>(ny, 0.0))
        {}

    void initializeU(
        const std::vector<std::vector<std::vector<double>>>& UInit
    ); 

    void oneStepRK2();

    void save(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    std::vector<std::vector<std::vector<double>>> getU();

    void calculateDt();

    bool checkCalculationIsCrashed();

private:
    void shiftUToCenterForCT(
        std::vector<std::vector<std::vector<double>>>& U
    );
    void backUToCenterHalfForCT(
        std::vector<std::vector<std::vector<double>>>& U
    );
};



