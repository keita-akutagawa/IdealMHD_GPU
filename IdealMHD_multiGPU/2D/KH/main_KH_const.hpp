#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../IdealMHD2D_GPU/const.hpp"
#include "../IdealMHD2D_GPU/idealMHD2D.hpp"
#include <mpi.h>


std::string directoryname = "results_KH";
std::string filenameWithoutStep = "KH";
std::ofstream logfile("log_KH.txt");

const int buffer = 3;

const double EPS = 1e-20;
const double PI = 3.141592653589793;

const double gamma_mhd = 5.0 / 3.0;

const double shear_thickness = 1.0;
const double rr = 0.2;
const double br = 1.0;
const double beta = 2.0;
const double theta = PI / 2.0;
const double rho0 = 1.0;
const double b0 = 1.0;
const double p0 = beta * b0 * b0 / 2.0;
const double v0 = sqrt(b0 * b0 / rho0 + gamma_mhd * p0 / rho0);

const double xmin = 0.0;
const double xmax = 2.0 * PI * shear_thickness / 0.4;
const double dx = shear_thickness / 32.0;
const int nx = int((xmax - xmin) / dx);
const double ymin = 0.0;
const double ymax = 2.0 * 10.0 * shear_thickness;
const double dy = shear_thickness / 32.0;
const int ny = int((ymax - ymin) / dy);

const double xCenter = (xmax - xmin) / 2.0;
const double yCenter = (ymax - ymin) / 2.0;

const double CFL = 0.7;
double dt = 0.0;
const int totalStep = 30000;
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

__constant__ double device_xCenter;
__constant__ double device_yCenter;

__constant__ double device_CFL;
__constant__ double device_gamma_mhd;

__device__ double device_dt;

__constant__ double device_shear_thickness;
__constant__ double device_rr;
__constant__ double device_br;
__constant__ double device_beta;
__constant__ double device_theta;
__constant__ double device_rho0;
__constant__ double device_b0;
__constant__ double device_p0;
__constant__ double device_v0;

