#pragma once
#include "matrixcsr.h"
#include "vector.h"
#include <vector>
#include <cstdint>

class Solver 
{
    public:
        // Static CPU solver
        static Vector solve_cg_cpu(
            const MatrixCSR& A,
            const Vector& b,
            float tol = 1e-6f,
            int max_iter = 1000);

        // Static GPU solver (CUDA: cuSPARSE + cuBLAS)
        static Vector solve_cg_gpu(
            const MatrixCSR& A,
            const Vector& b,
            float tol = 1e-6f,
            int max_iter = 1000);

        // Quasistatic Transient Solver (Eddy Currents) 100% VRAM resident (GPU)
        static void solve_transient_eddy_gpu(
            const MatrixCSR& K_eff,
            const MatrixCSR& M,
            const Vector& f0,
            float dt,
            int num_steps,
            float freq,
            int save_every,
            float* out_snapshots_flat,
            float tol = 1e-6f,
            int max_iter = 1000);

        // Quasistatic Transient Solver COUPLED WITH BARKHAUSEN MODEL (Exact ABBM) on GPU
        // Fully parameterized from Python without arbitrary constants in the backend.
        static void solve_transient_barkhausen_gpu(
            const MatrixCSR& K_eff,
            const MatrixCSR& M,
            const Vector& f0,
            const int32_t* elements,
            size_t num_elements,
            const int32_t* pids,
            int32_t target_pid,
            const float* sensor_bounds,
            const float* nodes_x,
            const float* nodes_y,
            const float* nodes_z,
            const float* inv_mu_per_elem,
            const float* sigma_per_elem,
            float mu0,
            float dt,
            int num_steps,
            float freq,
            int save_every,
            int num_domains,
            float G_const,
            float h_param,
            float Msat,
            float mean_hc,
            float std_hc,
            float h_anh,
            float corr_length,
            float avalanche_intensity,
            float alpha,
            const float* mag_axis,
            uint32_t seed,
            float* out_snapshots_flat,
            float* out_B_history,
            float* out_M_history,
            float* out_dMdt_history,
            float tol = 1e-6f,
            int max_iter = 1000);
};