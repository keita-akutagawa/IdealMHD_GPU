#include <thrust/device_vector.h>
#include "const.hpp"
#include "muscl.hpp"
#include "conservation_parameter_struct.hpp"
#include "basic_parameter_struct.hpp"
#include "mpi.hpp"


class CalculateQ
{
private:
    MPIInfo mPIInfo;
    MUSCL muscl;

public:
    CalculateQ(MPIInfo& mPIInfo);

    void setPhysicalParameterX(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<BasicParameter>& dQCenter
    );

    void setPhysicalParameterY(
        const thrust::device_vector<ConservationParameter>& U, 
        thrust::device_vector<BasicParameter>& dQCenter
    );

    void calculateLeftQX(
        const thrust::device_vector<BasicParameter>& dQCenter, 
        thrust::device_vector<BasicParameter>& dQLeft
    );

    void calculateLeftQY(
        const thrust::device_vector<BasicParameter>& dQCenter, 
        thrust::device_vector<BasicParameter>& dQLeft
    );

    void calculateRightQX(
        const thrust::device_vector<BasicParameter>& dQCenter, 
        thrust::device_vector<BasicParameter>& dQRight
    );

    void calculateRightQY(
        const thrust::device_vector<BasicParameter>& dQCenter, 
        thrust::device_vector<BasicParameter>& dQRight
    );
};

