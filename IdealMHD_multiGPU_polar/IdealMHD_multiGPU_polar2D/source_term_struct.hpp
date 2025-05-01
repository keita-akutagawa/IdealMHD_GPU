#ifndef SourceTerm_STRUCT_H
#define SourceTerm_STRUCT_H


struct SourceTerm
{
    double s0;
    double s1;
    double s2;
    double s3;
    double s4;
    double s5;
    double s6;
    double s7;

    __host__ __device__
    SourceTerm() : 
        s0(0.0), 
        s1(0.0),
        s2(0.0),
        s3(0.0),
        s4(0.0),
        s5(0.0),
        s6(0.0),
        s7(0.0)
        {}
};

#endif
