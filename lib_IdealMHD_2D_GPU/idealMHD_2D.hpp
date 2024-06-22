#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include "flux_solver.hpp"
#include "ct.hpp"
#include "boundary.hpp"


class IdealMHD2D
{
private:
    FluxSolver fluxSolver;
    thrust::device_vector<Flux> fluxF;
    thrust::device_vector<Flux> fluxG;
    thrust::device_vector<ConservationParameter> U;
    thrust::device_vector<ConservationParameter> UBar;
    thrust::device_vector<double> dtVector;
    Boundary boundary;
    CT ct;
    thrust::device_vector<double> bXOld;
    thrust::device_vector<double> bYOld;
    thrust::device_vector<double> tmpVector;
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

private:
    void shiftUToCenterForCT(
        thrust::device_vector<ConservationParameter>& U
    );

    void backUToCenterHalfForCT(
        thrust::device_vector<ConservationParameter>& U
    );
};



