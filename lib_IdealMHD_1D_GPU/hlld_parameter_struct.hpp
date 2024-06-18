#ifndef HLLD_PARAMETER_STRUCT_H
#define HLLD_PARAMETER_STRUCT_H


struct HLLDParameter
{
    double pT;
    double pT1;
    double e;
    double cs;
    double ca;
    double va;
    double cf;
    double S;
    double S1;
    double SM;

    __host__ __device__
    HLLDParameter() :
        pT(0.0), 
        pT1(0.0), 
        e(0.0), 
        cs(0.0), 
        ca(0.0), 
        va(0.0), 
        cf(0.0), 
        S(0.0), 
        S1(0.0), 
        SM(0.0)
        {}
};

#endif
