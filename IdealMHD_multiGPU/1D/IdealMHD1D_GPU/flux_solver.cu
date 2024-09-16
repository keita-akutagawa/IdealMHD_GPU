#include "flux_solver.hpp"


FluxSolver::FluxSolver(int localSize)
    : hLLD(localSize)
{
}


thrust::device_vector<Flux>& FluxSolver::getFluxF(
    const thrust::device_vector<ConservationParameter>& U
)
{
    hLLD.calculateFlux(U);
    flux = hLLD.getFlux();

    return flux;
}

