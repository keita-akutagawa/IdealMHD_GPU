#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "flux_struct.hpp"
#include "sign.hpp"
#include "mpi.hpp"


class CT
{
private:
    MPIInfo mPIInfo; 

    thrust::device_vector<double> oldNumericalFluxF_f5;
    thrust::device_vector<double> oldNumericalFluxG_f4;
    thrust::device_vector<double> oldFluxF_f5;
    thrust::device_vector<double> oldFluxG_f4;
    thrust::device_vector<double> oldNumericalFluxF_f0;
    thrust::device_vector<double> oldNumericalFluxG_f0;

    thrust::device_vector<double> nowNumericalFluxF_f5;
    thrust::device_vector<double> nowNumericalFluxG_f4;
    thrust::device_vector<double> nowFluxF_f5;
    thrust::device_vector<double> nowFluxG_f4;
    thrust::device_vector<double> nowNumericalFluxF_f0;
    thrust::device_vector<double> nowNumericalFluxG_f0;

    thrust::device_vector<double> eZVector;

public:
    CT(MPIInfo& mPIInfo);

    void setOldFlux2D( 
        const thrust::device_vector<Flux>& fluxF, 
        const thrust::device_vector<Flux>& fluxG, 
        const thrust::device_vector<ConservationParameter>& U
    );

    void setNowFlux2D( 
        const thrust::device_vector<Flux>& fluxF, 
        const thrust::device_vector<Flux>& fluxG, 
        const thrust::device_vector<ConservationParameter>& UBar
    );
    
    void divBClean( 
        const thrust::device_vector<double>& bXOld, 
        const thrust::device_vector<double>& bYOld, 
        thrust::device_vector<ConservationParameter>& U
    );
};

