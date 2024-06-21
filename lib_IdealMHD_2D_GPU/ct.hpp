#include <thrust/device_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include "flux_struct.hpp"


class CT
{
private:
    thrust::device_vector<double> EzVector;
    thrust::device_vector<Flux> oldFluxF;
    thrust::device_vector<Flux> oldFluxG;

public:

    void setOldFlux2D( 
        const thrust::device_vector<Flux>& fluxF, 
        const thrust::device_vector<Flux>& fluxG
    );
    
    void divBClean( 
        thrust::device_vector<ConservationParameter>& U
    );
};

