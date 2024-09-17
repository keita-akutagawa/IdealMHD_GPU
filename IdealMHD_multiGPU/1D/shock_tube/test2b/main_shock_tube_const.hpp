#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../../IdealMHD1D_GPU/const.hpp"
#include "../../IdealMHD1D_GPU/idealMHD1D.hpp"
#include "../../IdealMHD1D_GPU/mpi.hpp"



std::string directoryname = "/cfca-work/akutagawakt/IdealMHD_multiGPU/results_shock_tube_test2b";
std::string filenameWithoutStep = "shock_tube";
std::ofstream logfile("/cfca-work/akutagawakt/IdealMHD_multiGPU/results_shock_tube_test2b/log_shock_tube.txt");

const int buffer = 2;

const double EPS = 1e-20;
const double PI = 3.141592653589793;
const double dx = 0.001;
const double xmin = 0.0;
const double xmax = 1.0;
const int nx = int((xmax - xmin) / dx);
const double CFL = 0.7;
const double gamma_mhd = 5.0 / 3.0;
double dt = 0.0;
const int totalStep = 1000;
const int recordStep = 100;
double totalTime = 0.0;


__constant__ double device_EPS;
__constant__ double device_PI;

__constant__ double device_dx;
__constant__ double device_xmin;
__constant__ double device_xmax;
__constant__ int device_nx;

__constant__ double device_CFL;
__constant__ double device_gamma_mhd;

__device__ double device_dt;

__constant__ int device_totalStep;
__device__ double device_totalTime;

