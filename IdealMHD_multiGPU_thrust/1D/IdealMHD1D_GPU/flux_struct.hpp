#ifndef Flux_STRUCT_H
#define Flux_STRUCT_H


struct Flux
{
    double f0;
    double f1;
    double f2;
    double f3;
    double f4;
    double f5;
    double f6;
    double f7;

    __host__ __device__
    Flux() : 
        f0(0.0), 
        f1(0.0),
        f2(0.0),
        f3(0.0),
        f4(0.0),
        f5(0.0),
        f6(0.0),
        f7(0.0)
        {}
};

#endif
