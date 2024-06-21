#include <thrust/device_vector.h>
#include "const.hpp"
#include "flux_struct.hpp"


class CT
{
private:
    thrust::device_vector<double> EzVector;
    thrust::device_vector<Flux> oldFlux2D;

public:

    void setOldFlux2D( 
        const thrust::device_vector<Flux>& flux2D
    );
    
    void divBClean( 
        const std::vector<std::vector<double>>& bxOld,
        const std::vector<std::vector<double>>& byOld, 
        std::vector<std::vector<std::vector<double>>>& U
    );
};

