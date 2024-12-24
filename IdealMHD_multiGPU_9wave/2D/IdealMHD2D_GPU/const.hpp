#ifndef CONST_STRUCT_H
#define CONST_STRUCT_H

extern const double EPS;
extern const double PI;

extern const double dx;
extern const double xmin;
extern const double xmax;
extern const int nx;

extern const double dy;
extern const double ymin;
extern const double ymax;
extern const int ny;

extern const double CFL;
extern const double gamma_mhd;

extern double dt;

extern double ch; 
extern double cp; 
extern double cr; 

extern const int totalStep;
extern double totalTime;


extern __constant__ double device_EPS;
extern __constant__ double device_PI;

extern __constant__ double device_dx;
extern __constant__ double device_xmin;
extern __constant__ double device_xmax;
extern __constant__ int device_nx;

extern __constant__ double device_dy;
extern __constant__ double device_ymin;
extern __constant__ double device_ymax;
extern __constant__ int device_ny;

extern __constant__ double device_CFL;
extern __constant__ double device_gamma_mhd;

extern __device__ double device_dt;

extern __device__ double device_ch; 
extern __device__ double device_cp; 
extern __device__ double device_cr; 


void initializeDeviceConstants();

#endif


