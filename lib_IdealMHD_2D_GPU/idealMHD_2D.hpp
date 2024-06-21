#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include "flux_solver.hpp"
#include "boundary.hpp"


class IdealMHD2D
{
private:
    FluxSolver fluxSolverF;
    FluxSolver fluxSolverG;
    thrust::device_vector<Flux> fluxF;
    thrust::device_vector<Flux> fluxG;
    thrust::device_vector<ConservationParameter> U;
    thrust::device_vector<ConservationParameter> UBar;
    thrust::device_vector<double> dtVector;
    Boundary boundary;
    thrust::host_vector<ConservationParameter> hU;

public:
    IdealMHD2D();

    virtual void initializeU(); 

    void oneStepRK2();

    void save(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    thrust::device_vector<ConservationParameter> getU();

    void calculateDt();

    bool checkCalculationIsCrashed();
};



