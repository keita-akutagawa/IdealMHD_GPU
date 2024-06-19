#ifndef BASIC_PARAMETER_STRUCT_H
#define BASIC_PARAMETER_STRUCT_H


struct BasicParameter
{
    double rho;
    double u;
    double v;
    double w;
    double bX; 
    double bY;
    double bZ;
    double p;

    __device__
    BasicParameter() : 
        rho(0.0), 
        u(0.0), 
        v(0.0), 
        w(0.0), 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0), 
        p(0.0)
        {}
};

#endif
