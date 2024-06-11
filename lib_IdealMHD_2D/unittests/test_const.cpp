#include "../const.hpp"

const double EPS = 1e-20;

const double dx = 0.01;
const double xmin = 0.0;
const double xmax = 1.0;
const int nx = int((xmax - xmin) / dx);

const double CFL = 0.7;
const double gamma_mhd = 5.0 / 3.0;

double dt = 0.0;

const int totalStep = 100;

