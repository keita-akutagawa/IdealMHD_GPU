#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#include "../../IdealMHD_multiGPU_polar2D/idealMHD2D.hpp"
#include "../../IdealMHD_multiGPU_polar2D/const.hpp"


std::string directoryName = "/cfca-work/akutagawakt/IdealMHD_multiGPU_polar/results_steady";
std::string filenameWithoutStep = "steady";
std::ofstream logfile(directoryName + "/log_steady.txt"   );
std::ofstream mpifile(directoryName + "/mpilog_steady.txt");


const int buffer = 3; 

const int totalStep = 1000;
const int recordStep = 10;

double totalTime = 0.0;

const double EPS = 1e-20;
const double PI = 3.14159265358979;

double eta = 0.0;
double viscosity = 0.0;

const int nx = 100;
const double dx = 0.1;
const double xmin = 10.0 * dx;
const double xmax = nx * dx + xmin;

const int ny = 90;
const double dy = 2.0 * PI / ny;
const double ymin = -PI;
const double ymax = PI;

const double CFL = 0.7;
const double gamma_mhd = 5.0 / 3.0;

double dt = 0.0;


////////// device //////////

__constant__ double device_EPS;
__constant__ double device_PI;

__device__ double device_eta; 
__device__ double device_viscosity; 

__constant__ double device_dx;
__constant__ double device_xmin;
__constant__ double device_xmax;
__constant__ int device_nx;

__constant__ double device_dy;
__constant__ double device_ymin;
__constant__ double device_ymax;
__constant__ int device_ny;

__constant__ double device_CFL;
__constant__ double device_gamma_mhd;

__device__ double device_dt;

