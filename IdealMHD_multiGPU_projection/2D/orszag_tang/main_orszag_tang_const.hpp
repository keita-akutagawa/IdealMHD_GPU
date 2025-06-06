#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../IdealMHD2D_GPU/const.hpp"
#include "../IdealMHD2D_GPU/idealMHD2D.hpp"

/* standard or dynamically load library */
#ifdef AMGX_DYNAMIC_LOADING
#include "amgx_capi.h"
#else
#include "amgx_c.h"
#endif


std::string directoryname = "/cfca-work/akutagawakt/IdealMHD_multiGPU_projection/results_orszag_tang";
std::string filenameWithoutStep = "orszag_tang";
std::ofstream logfile(directoryname + "/log_orszag_tang.txt");

const int buffer = 3;

const std::string MTXfilename = "/home/akutagawakt/IdealMHD_multiGPU_projection/2D/orszag_tang/poisson_periodic.mtx";
const std::string jsonFilenameForSolver = "/home/akutagawakt/IdealMHD_multiGPU_projection/2D/IdealMHD2D_GPU/AMG_CLASSICAL_CG.json";

const double EPS = 1.0e-20;
const double PI = 3.141592653589793;

const int nx = 256;
const double xmin = 0.0;
const double xmax = 2.0 * PI;
const double dx = (xmax - xmin) / nx;
const int ny = 256;
const double ymin = 0.0;
const double ymax = 2.0 * PI;
const double dy = (ymax - ymin) / ny;

const double CFL = 0.7;
const double gamma_mhd = 5.0 / 3.0;
double dt = 0.0;

const int totalStep = 10000;
const int recordStep = 100;
double totalTime = 0.0;

__constant__ double device_EPS;
__constant__ double device_PI;

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

