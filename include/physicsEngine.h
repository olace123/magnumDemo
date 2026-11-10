#pragma once

#include "matrixcsr.h"
#include "vector.h"
#include <cstdint>
#include <cstddef>

struct MaterialSpec {
    int32_t id;
    float mu_r;
    float sigma;
};

class PhysicsEngine 
{
public:
    /**
     * @brief Assigns 1/mu_r and conductivity sigma (optional, nullptr in magnetostatic regime)
     *        for an arbitrary number of regions using a Lookup Table (LUT) in O(1) time.
     */
    static void setup_materials(
        const int32_t* pids, size_t num_elements,
        float* inv_mu_per_elem,
        float* sigma_per_elem,
        const MaterialSpec* materials, size_t num_materials,
        float default_mu_r = 1.0f, float default_sigma = 0.0f);

    /**
     * @brief Assigns azimuthal current density J in a cylindrical coil
     *        around the specified axis (axis: 0=X, 1=Y, 2=Z).
     *        - If axis=0 (X): center_u = Y_c, center_v = Z_c
     *        - If axis=1 (Y): center_u = Z_c, center_v = X_c
     *        - If axis=2 (Z): center_u = X_c, center_v = Y_c
     */
    static void setup_coil_source(
        const int32_t* elements, size_t num_elements,
        const int32_t* pids,
        const float* nodes_x, const float* nodes_y, const float* nodes_z,
        float* Jx_per_elem, float* Jy_per_elem, float* Jz_per_elem,
        int32_t id_coil, float j_mag,
        float center_u, float center_v, int axis = 0, bool clear_first = true);

    /**
     * @brief Static assembly (K and f).
     */
    static void assemble_physics(MatrixCSR& K, Vector& f,
        const int32_t* elements, size_t num_elements,
        const float* nodes_x, const float* nodes_y, const float* nodes_z,
        const float* inv_mu_per_elem,
        const float* Jx_per_elem, const float* Jy_per_elem, const float* Jz_per_elem,
        float mu0);

    /**
     * @brief Quasistatic transient assembly:
     *        Builds K_eff = K + (1/dt)*M, the mass matrix M, and the source amplitude vector f0.
     */
    static void assemble_transient_system(
        MatrixCSR& K_eff, MatrixCSR& M, Vector& f0,
        const int32_t* elements, size_t num_elements,
        const float* nodes_x, const float* nodes_y, const float* nodes_z,
        const float* inv_mu_per_elem,
        const float* sigma_per_elem,
        const float* Jx_per_elem, const float* Jy_per_elem, const float* Jz_per_elem,
        float mu0, float dt);

    /**
     * @brief Automatically detects the bounding box and applies the Dirichlet penalty.
     */
    static void apply_dirichlet_bounding_box(MatrixCSR& K,
        const float* nodes_x, const float* nodes_y, const float* nodes_z,
        size_t num_nodes, float tol, float penalty);

    /**
     * @brief Computes B = curl(A) and |B| for each tetrahedron.
     */
    static void compute_b_field(
        const int32_t* elements, size_t num_elements,
        const float* nodes_x, const float* nodes_y, const float* nodes_z,
        const float* A_vec,
        float* Bx_elem, float* By_elem, float* Bz_elem, float* Bmag_elem);

    /**
     * @brief Computes the induced eddy current density J_eddy = -sigma * (A_curr - A_prev) / dt in each tetrahedron.
     */
    static void compute_eddy_current(
        const int32_t* elements, size_t num_elements,
        const float* sigma_per_elem,
        const float* A_curr, const float* A_prev, float dt,
        float* J_eddy_x, float* J_eddy_y, float* J_eddy_z, float* J_eddy_mag);
};