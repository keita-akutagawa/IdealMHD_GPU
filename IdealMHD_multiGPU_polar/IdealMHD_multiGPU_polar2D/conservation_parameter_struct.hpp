#ifndef CONSERVATION_PARAMETER_STRUCT_H
#define CONSERVATION_PARAMETER_STRUCT_H


struct ConservationParameter
{
    double rho;
    double rhoU;
    double rhoV;
    double rhoW;
    double bX; 
    double bY;
    double bZ;
    double e;

    __host__ __device__
    ConservationParameter() : 
        rho(0.0), 
        rhoU(0.0), 
        rhoV(0.0), 
        rhoW(0.0), 
        bX(0.0), 
        bY(0.0), 
        bZ(0.0), 
        e(0.0)
        {}
    
    __host__ __device__
    ConservationParameter(double rho, double rhoU, double rhoV, double rhoW, 
                          double bX, double bY, double bZ, double e) :
        rho(rho), 
        rhoU(rhoU),
        rhoV(rhoV), 
        rhoW(rhoW), 
        bX(bX), 
        bY(bY), 
        bZ(bZ), 
        e(e)
    {}
    
    __host__ __device__
    ConservationParameter operator+(const ConservationParameter& other) const
    {
        return ConservationParameter(rho + other.rho, rhoU + other.rhoU, rhoV + other.rhoV, rhoW + other.rhoW, 
                                     bX + other.bX, bY + other.bY, bZ + other.bZ, e + other.e);
    }
    
    __host__ __device__
    ConservationParameter& operator+=(const ConservationParameter& other)
    {
        rho  += other.rho;
        rhoU += other.rhoU;
        rhoV += other.rhoV;
        rhoW += other.rhoW;
        bX   += other.bX;
        bY   += other.bY;
        bZ   += other.bZ;
        e    += other.e;
        
        return *this;
    }

    __host__ __device__
    ConservationParameter operator*(double scalar) const
    {
        return ConservationParameter(scalar * rho, scalar * rhoU, scalar * rhoV, scalar * rhoW, 
                                     scalar * bX, scalar * bY, scalar * bZ, scalar * e);
    }

    __host__ __device__
    friend ConservationParameter operator*(double scalar, const ConservationParameter& other) 
    {
        return ConservationParameter(scalar * other.rho, scalar * other.rhoU, scalar * other.rhoV, scalar * other.rhoW, 
                                     scalar * other.bX, scalar * other.bY, scalar * other.bZ, scalar * other.e);
    }

    __host__ __device__
    ConservationParameter operator/(double scalar) const
    {
        return ConservationParameter(rho / scalar, rhoU / scalar, rhoV / scalar, rhoW / scalar,
                                    bX / scalar, bY / scalar, bZ / scalar, e / scalar);
    }
};

#endif
