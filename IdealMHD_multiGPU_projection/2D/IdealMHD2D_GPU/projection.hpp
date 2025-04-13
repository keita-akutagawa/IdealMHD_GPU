#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "const.hpp"
#include "conservation_parameter_struct.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "mpi.hpp"
#include "amgx_config.h"


// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause]]]
//
//modified by Keita Akutagawa [2025.4.10]
//

/* standard or dynamically load library */
#ifdef AMGX_DYNAMIC_LOADING
#include "amgx_capi.h"
#else
#include "amgx_c.h"
#endif


class Projection
{
private: 
    MPIInfo mPIInfo;

    thrust::device_vector<double> divB; 
    thrust::device_vector<double> sum_divB; 
    thrust::device_vector<double> psi; 

    AMGX_config_handle config;
    AMGX_resources_handle resource;
    AMGX_matrix_handle A;
    AMGX_vector_handle amgx_sol, amgx_rhs;
    AMGX_solver_handle solver;


public: 
    Projection(MPIInfo& mPIInfo); 

    ~Projection(); 

    void correctB(
        thrust::device_vector<ConservationParameter>& U
    ); 

private:

};


