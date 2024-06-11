#include "const.hpp"
#include "hlld.hpp"


class HLLD;

class FluxSolver
{
private:
    HLLD hLLD;
    Flux flux;

public:
    Flux getFluxF(
        const std::vector<std::vector<double>> U
    );
};


