#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include "flux_solver.hpp"
#include "boundary.hpp"


class IdealMHD1D
{
private:
    FluxSolver fluxSolver;
    thrust::device_vector<Flux> fluxF;
    thrust::device_vector<ConservationParameter> U;
    thrust::device_vector<ConservationParameter> UBar;
    Boundary boundary;

public:
    IdealMHD1D() : 
        fluxF(device_nx), 
        U(device_nx), 
        UBar(device_nx)
        {}

    virtual void initializeU(); 

    void oneStepRK2();

    void save(
        std::string directoryname, 
        std::string filenameWithoutStep, 
        int step
    );

    thrust::device_vector<ConservationParameter> getU();

    void calculateDt();
};



