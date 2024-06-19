#ifndef HLLD_PARAMETER_STRUCT_H
#define HLLD_PARAMETER_STRUCT_H


struct HLLDParameter
{
    double pTL;
    double pTR;
    double eL;
    double eR;
    double csL;
    double csR;
    double caL;
    double caR;
    double vaL;
    double vaR;
    double cfL;
    double cfR;

    double SL;
    double SR;
    double SM;

    double rho1L;
    double rho1R;
    double u1L;
    double u1R;
    double v1L;
    double v1R;
    double w1L;
    double w1R;
    double bY1L;
    double bY1R;
    double bZ1L;
    double bZ1R;
    double e1L;
    double e1R; 
    double pT1L;
    double pT1R;

    double S1L;
    double S1R;

    double rho2L;
    double rho2R;
    double u2;
    double v2;
    double w2; 
    double bY2; 
    double bZ2;
    double e2L;
    double e2R;
    double pT2L;
    double pT2R;
    

    __device__
    HLLDParameter() :
        pTL(0.0),
        pTR(0.0),
        eL(0.0),
        eR(0.0),
        csL(0.0),
        csR(0.0),
        caL(0.0),
        caR(0.0),
        vaL(0.0),
        vaR(0.0),
        cfL(0.0),
        cfR(0.0),

        SL(0.0),
        SR(0.0),
        SM(0.0),

        rho1L(0.0),
        rho1R(0.0),
        u1L(0.0),
        u1R(0.0),
        v1L(0.0),
        v1R(0.0),
        w1L(0.0),
        w1R(0.0),
        bY1L(0.0),
        bY1R(0.0),
        bZ1L(0.0),
        bZ1R(0.0),
        e1L(0.0),
        e1R(0.0), 
        pT1L(0.0),
        pT1R(0.0),

        S1L(0.0),
        S1R(0.0),

        rho2L(0.0),
        rho2R(0.0),
        u2(0.0),
        v2(0.0),
        w2(0.0), 
        bY2(0.0), 
        bZ2(0.0),
        e2L(0.0),
        e2R(0.0),
        pT2L(0.0),
        pT2R(0.0)
        {}
};

#endif
