#include "muscl.hpp"
#include <thrust/transform.h>
#include <thrust/tuple.h>


MUSCL::MUSCL(MPIInfo& mPIInfo)
    : mPIInfo(mPIInfo)
{
}


__global__ void leftParameter_kernel(
    const BasicParameter* dQ, 
    BasicParameter* dQLeft, 
    int localSizeX, int localSizeY, int shiftForNeighbor
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < i) && (i < localSizeX - 1) && (0 < j) &&  (j < localSizeY - 1)) {
        int index = j + i * localSizeY;

        dQLeft[index].rho = dQ[index].rho + 0.5 * minmod(dQ[index].rho - dQ[index - shiftForNeighbor].rho, dQ[index + shiftForNeighbor].rho - dQ[index].rho);
        dQLeft[index].u   = dQ[index].u   + 0.5 * minmod(dQ[index].u   - dQ[index - shiftForNeighbor].u  , dQ[index + shiftForNeighbor].u   - dQ[index].u  );
        dQLeft[index].v   = dQ[index].v   + 0.5 * minmod(dQ[index].v   - dQ[index - shiftForNeighbor].v  , dQ[index + shiftForNeighbor].v   - dQ[index].v  );
        dQLeft[index].w   = dQ[index].w   + 0.5 * minmod(dQ[index].w   - dQ[index - shiftForNeighbor].w  , dQ[index + shiftForNeighbor].w   - dQ[index].w  );
        dQLeft[index].bX  = dQ[index].bX  + 0.5 * minmod(dQ[index].bX  - dQ[index - shiftForNeighbor].bX , dQ[index + shiftForNeighbor].bX  - dQ[index].bX );
        dQLeft[index].bY  = dQ[index].bY  + 0.5 * minmod(dQ[index].bY  - dQ[index - shiftForNeighbor].bY , dQ[index + shiftForNeighbor].bY  - dQ[index].bY );
        dQLeft[index].bZ  = dQ[index].bZ  + 0.5 * minmod(dQ[index].bZ  - dQ[index - shiftForNeighbor].bZ , dQ[index + shiftForNeighbor].bZ  - dQ[index].bZ );
        dQLeft[index].p   = dQ[index].p   + 0.5 * minmod(dQ[index].p   - dQ[index - shiftForNeighbor].p  , dQ[index + shiftForNeighbor].p   - dQ[index].p  );
        dQLeft[index].psi = dQ[index].psi + 0.5 * minmod(dQ[index].psi - dQ[index - shiftForNeighbor].psi, dQ[index + shiftForNeighbor].psi - dQ[index].psi);
    }
}


void MUSCL::getLeftQX(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    leftParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQ.data()), 
        thrust::raw_pointer_cast(dQLeft.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, mPIInfo.localSizeY
    );
}


void MUSCL::getLeftQY(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQLeft
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    leftParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQ.data()), 
        thrust::raw_pointer_cast(dQLeft.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 1
    );
}


__global__ void rightParameter_kernel(
    const BasicParameter* dQ, 
    BasicParameter* dQRight, 
    int localSizeX, int localSizeY, int shiftForNeighbor
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localSizeX - 2 && j < localSizeY - 2) {
        int index = j + i * localSizeY;

        dQRight[index].rho = dQ[index + shiftForNeighbor].rho - 0.5 * minmod(dQ[index + shiftForNeighbor].rho - dQ[index].rho, dQ[index + 2 * shiftForNeighbor].rho - dQ[index + shiftForNeighbor].rho);
        dQRight[index].u   = dQ[index + shiftForNeighbor].u   - 0.5 * minmod(dQ[index + shiftForNeighbor].u   - dQ[index].u  , dQ[index + 2 * shiftForNeighbor].u   - dQ[index + shiftForNeighbor].u  );
        dQRight[index].v   = dQ[index + shiftForNeighbor].v   - 0.5 * minmod(dQ[index + shiftForNeighbor].v   - dQ[index].v  , dQ[index + 2 * shiftForNeighbor].v   - dQ[index + shiftForNeighbor].v  );
        dQRight[index].w   = dQ[index + shiftForNeighbor].w   - 0.5 * minmod(dQ[index + shiftForNeighbor].w   - dQ[index].w  , dQ[index + 2 * shiftForNeighbor].w   - dQ[index + shiftForNeighbor].w  );
        dQRight[index].bX  = dQ[index + shiftForNeighbor].bX  - 0.5 * minmod(dQ[index + shiftForNeighbor].bX  - dQ[index].bX , dQ[index + 2 * shiftForNeighbor].bX  - dQ[index + shiftForNeighbor].bX );
        dQRight[index].bY  = dQ[index + shiftForNeighbor].bY  - 0.5 * minmod(dQ[index + shiftForNeighbor].bY  - dQ[index].bY , dQ[index + 2 * shiftForNeighbor].bY  - dQ[index + shiftForNeighbor].bY );
        dQRight[index].bZ  = dQ[index + shiftForNeighbor].bZ  - 0.5 * minmod(dQ[index + shiftForNeighbor].bZ  - dQ[index].bZ , dQ[index + 2 * shiftForNeighbor].bZ  - dQ[index + shiftForNeighbor].bZ );
        dQRight[index].p   = dQ[index + shiftForNeighbor].p   - 0.5 * minmod(dQ[index + shiftForNeighbor].p   - dQ[index].p  , dQ[index + 2 * shiftForNeighbor].p   - dQ[index + shiftForNeighbor].p  );
        dQRight[index].psi = dQ[index + shiftForNeighbor].psi - 0.5 * minmod(dQ[index + shiftForNeighbor].psi - dQ[index].psi, dQ[index + 2 * shiftForNeighbor].psi - dQ[index + shiftForNeighbor].psi);
    }
}


void MUSCL::getRightQX(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQRight
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    rightParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQ.data()), 
        thrust::raw_pointer_cast(dQRight.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, mPIInfo.localSizeY
    );
}


void MUSCL::getRightQY(
    const thrust::device_vector<BasicParameter>& dQ, 
    thrust::device_vector<BasicParameter>& dQRight
)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((mPIInfo.localSizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (mPIInfo.localSizeY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    rightParameter_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(dQ.data()), 
        thrust::raw_pointer_cast(dQRight.data()), 
        mPIInfo.localSizeX, mPIInfo.localSizeY, 1
    );
}

