#ifndef FAN_PARAMETER_STRUCT_H
#define FAN_PARAMETER_STRUCT_H


struct FanParameter
{
    double rho;
    double u;
    double v; 
    double w;
    double bX;
    double bY; 
    double bZ;
    double e;
    double pT;

    __host__ __device__ 
    FanParameter() :
        rho(0.0), 
        u(0.0), 
        v(0.0), 
        w(0.0), 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0),
        e(0.0), 
        pT(0.0)
        {}
};

#endif
