#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrixcsr.h"
#include "solvers.h"
#include "physicsEngine.h"
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// BRIDGE 1: Topological Assembly
// -----------------------------------------------------------------------------
void assemble_topology_csr(MatrixCSR& mat, py::array_t<int32_t> rows_np, py::array_t<int32_t> cols_np) 
{
    auto buf_rows = rows_np.request();
    auto buf_cols = cols_np.request();

    if (buf_rows.shape[0] != buf_cols.shape[0]) {
        throw std::invalid_argument("Row and column arrays must have the same size.");
    }

    int32_t* ptr_rows = static_cast<int32_t*>(buf_rows.ptr);
    int32_t* ptr_cols = static_cast<int32_t*>(buf_cols.ptr);
    size_t num_connections = buf_rows.shape[0];

    std::vector<size_t> count_per_row(mat.rows(), 0); 
    for(size_t i = 0; i < num_connections; ++i) {
        count_per_row[ptr_rows[i]]++;
    }
    
    for(size_t i = 0; i < mat.rows(); ++i) {
        mat.count_node_conexions(i, count_per_row[i]);
    }

    mat.build_structure();

    for(size_t i = 0; i < num_connections; ++i) {
        mat.insert_topology(ptr_rows[i], ptr_cols[i]);
    }

    mat.finalize();
}

void assemble_topology_from_elements(MatrixCSR& mat, py::array_t<int32_t> elements_np)
{
    auto buf_elems = elements_np.request();
    const int32_t* elems = static_cast<const int32_t*>(buf_elems.ptr);
    size_t num_elements = buf_elems.shape[0] / 4;

    if (mat.rows() % 3 != 0) {
        throw std::invalid_argument("MatrixCSR dimensions must be a multiple of 3 (3 DOFs per node).");
    }
    size_t num_nodes = mat.rows() / 3;

    std::vector<std::vector<int32_t>> node_adj(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        node_adj[i].reserve(32);
    }

    for (size_t e = 0; e < num_elements; ++e) {
        const int32_t* n = elems + e * 4;
        for (int r = 0; r < 4; ++r) {
            int32_t nr = n[r];
            for (int c = 0; c < 4; ++c) {
                node_adj[nr].push_back(n[c]);
            }
        }
    }

    for (size_t nd = 0; nd < num_nodes; ++nd) {
        auto& neighbors = node_adj[nd];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        size_t nnz_per_dof = neighbors.size() * 3;
        mat.count_node_conexions(3 * nd + 0, nnz_per_dof);
        mat.count_node_conexions(3 * nd + 1, nnz_per_dof);
        mat.count_node_conexions(3 * nd + 2, nnz_per_dof);
    }

    mat.build_structure();

    for (size_t nd = 0; nd < num_nodes; ++nd) {
        const auto& neighbors = node_adj[nd];
        for (int k = 0; k < 3; ++k) {
            size_t row = 3 * nd + k;
            for (int32_t nb : neighbors) {
                mat.insert_topology(row, 3 * static_cast<size_t>(nb) + 0);
                mat.insert_topology(row, 3 * static_cast<size_t>(nb) + 1);
                mat.insert_topology(row, 3 * static_cast<size_t>(nb) + 2);
            }
        }
    }

    mat.finalize();
}

// -----------------------------------------------------------------------------
// BRIDGE 2: Solvers (Static, Transient Eddy & Transient Barkhausen in GPU / CPU)
// -----------------------------------------------------------------------------
py::array_t<float> solve_system_gpu(const MatrixCSR& A, py::array_t<float> b_np, float tol, int max_iter) 
{
    auto buf_b = b_np.request();
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    size_t size_b = buf_b.shape[0];

    if (size_b != A.rows()) {
        throw std::invalid_argument("Dimension of vector 'b' does not match matrix 'A'.");
    }

    const Vector b_cpp(ptr_b, size_b);
    Vector x_cpp = Solver::solve_cg_gpu(A, b_cpp, tol, max_iter);

    auto result = py::array_t<float>(size_b);
    auto buf_result = result.request();
    float* ptr_result = static_cast<float*>(buf_result.ptr);
    
    for(size_t i = 0; i < size_b; ++i) {
        ptr_result[i] = x_cpp[i];
    }

    return result;
}

py::array_t<float> solve_system_cpu(const MatrixCSR& A, py::array_t<float> b_np, float tol, int max_iter) 
{
    auto buf_b = b_np.request();
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    size_t size_b = buf_b.shape[0];

    if (size_b != A.rows()) {
        throw std::invalid_argument("Dimension of vector 'b' does not match matrix 'A'.");
    }

    const Vector b_cpp(ptr_b, size_b);
    Vector x_cpp = Solver::solve_cg_cpu(A, b_cpp, tol, max_iter);

    auto result = py::array_t<float>(size_b);
    auto buf_result = result.request();
    float* ptr_result = static_cast<float*>(buf_result.ptr);
    
    for(size_t i = 0; i < size_b; ++i) {
        ptr_result[i] = x_cpp[i];
    }

    return result;
}

py::array_t<float> solve_transient_eddy_gpu_bridge(
    const MatrixCSR& K_eff,
    const MatrixCSR& M,
    py::array_t<float> f0_np,
    float dt,
    int num_steps,
    float freq,
    int save_every,
    float tol,
    int max_iter)
{
    auto buf_f0 = f0_np.request();
    size_t N = buf_f0.shape[0];
    if (N != K_eff.rows() || N != M.rows()) {
        throw std::invalid_argument("Dimensions of K_eff, M, and f0 must match.");
    }

    const Vector f0_cpp(static_cast<float*>(buf_f0.ptr), N);
    int num_saved = (num_steps / save_every) + 1;

    py::array_t<float> snapshots({static_cast<size_t>(num_saved), N});
    float* ptr_out = static_cast<float*>(snapshots.request().ptr);

    Solver::solve_transient_eddy_gpu(
        K_eff, M, f0_cpp, dt, num_steps, freq, save_every, ptr_out, tol, max_iter);

    return snapshots;
}

py::tuple solve_transient_barkhausen_gpu_bridge(
    const MatrixCSR& K_eff,
    const MatrixCSR& M,
    py::array_t<float> f0_np,
    py::array_t<int32_t> elements_np,
    py::array_t<int32_t> pids_np,
    int32_t target_pid,
    py::object sensor_bounds_obj,
    py::array_t<float> nodes_x_np,
    py::array_t<float> nodes_y_np,
    py::array_t<float> nodes_z_np,
    py::array_t<float> inv_mu_np,
    py::array_t<float> sigma_np,
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
    py::object mag_axis_obj,
    uint32_t seed,
    float tol,
    int max_iter)
{
    auto buf_f0 = f0_np.request();
    size_t N = buf_f0.shape[0];
    auto buf_elems = elements_np.request();
    size_t num_elems = buf_elems.shape[0] / 4;

    float bounds_arr[6];
    const float* bounds_ptr = nullptr;
    if (!sensor_bounds_obj.is_none()) {
        auto seq = sensor_bounds_obj.cast<py::sequence>();
        if (seq.size() != 6) {
            throw std::invalid_argument("sensor_bounds debe ser una tupla/lista de 6 floats: (xmin, xmax, ymin, ymax, zmin, zmax) o None.");
        }
        for (int i = 0; i < 6; ++i) {
            bounds_arr[i] = seq[i].cast<float>();
        }
        bounds_ptr = bounds_arr;
    }

    float axis_arr[3] = {1.0f, 0.0f, 0.0f};
    if (!mag_axis_obj.is_none()) {
        auto seq = mag_axis_obj.cast<py::sequence>();
        if (seq.size() != 3) {
            throw std::invalid_argument("mag_axis debe ser una tupla/lista de 3 floats: (ux, uy, uz).");
        }
        for (int i = 0; i < 3; ++i) {
            axis_arr[i] = seq[i].cast<float>();
        }
    }

    const Vector f0_cpp(static_cast<float*>(buf_f0.ptr), N);
    int num_saved = (save_every > 0) ? ((num_steps / save_every) + 1) : 1;

    py::array_t<float> snapshots({static_cast<size_t>(num_saved), N});
    py::array_t<float> B_hist(static_cast<size_t>(num_steps + 1));
    py::array_t<float> M_hist(static_cast<size_t>(num_steps + 1));
    py::array_t<float> dMdt_hist(static_cast<size_t>(num_steps + 1));

    Solver::solve_transient_barkhausen_gpu(
        K_eff, M, f0_cpp,
        static_cast<const int32_t*>(buf_elems.ptr), num_elems,
        static_cast<const int32_t*>(pids_np.request().ptr), target_pid,
        bounds_ptr,
        static_cast<const float*>(nodes_x_np.request().ptr),
        static_cast<const float*>(nodes_y_np.request().ptr),
        static_cast<const float*>(nodes_z_np.request().ptr),
        static_cast<const float*>(inv_mu_np.request().ptr),
        static_cast<const float*>(sigma_np.request().ptr),
        mu0, dt, num_steps, freq, save_every,
        num_domains, G_const, h_param, Msat, mean_hc, std_hc,
        h_anh, corr_length, avalanche_intensity, alpha,
        axis_arr, seed,
        static_cast<float*>(snapshots.request().ptr),
        static_cast<float*>(B_hist.request().ptr),
        static_cast<float*>(M_hist.request().ptr),
        static_cast<float*>(dMdt_hist.request().ptr),
        tol, max_iter);

    return py::make_tuple(snapshots, B_hist, M_hist, dMdt_hist);
}

// -----------------------------------------------------------------------------
// BRIDGE 3: Physics Assembly & Pre/Post Processing
// -----------------------------------------------------------------------------
void setup_materials_bridge(
    py::array_t<int32_t> pids_np,
    py::array_t<float> inv_mu_np,
    py::object sigma_obj,
    py::args material_tuples,
    float default_mu_r,
    float default_sigma)
{
    auto buf_pids = pids_np.request();
    size_t num_elems = buf_pids.shape[0];
    float* sigma_ptr = nullptr;

    if (!sigma_obj.is_none()) {
        auto sigma_arr = sigma_obj.cast<py::array_t<float>>();
        sigma_ptr = static_cast<float*>(sigma_arr.request().ptr);
    }

    std::vector<MaterialSpec> specs;
    specs.reserve(material_tuples.size());

    for (auto item : material_tuples) {
        auto seq = item.cast<py::sequence>();
        if (seq.size() < 2) {
            throw std::invalid_argument("Cada material en *args debe ser una tupla (id, mu_r) o (id, mu_r, sigma).");
        }
        int32_t id = seq[0].cast<int32_t>();
        float mu_r = seq[1].cast<float>();
        float sig  = (seq.size() >= 3) ? seq[2].cast<float>() : default_sigma;
        specs.push_back({id, mu_r, sig});
    }

    PhysicsEngine::setup_materials(
        static_cast<const int32_t*>(buf_pids.ptr), num_elems,
        static_cast<float*>(inv_mu_np.request().ptr),
        sigma_ptr,
        specs.data(), specs.size(),
        default_mu_r, default_sigma);
}

void setup_coil_source_bridge(
    py::array_t<int32_t> elements_np,
    py::array_t<int32_t> pids_np,
    py::array_t<float> nodes_x_np, py::array_t<float> nodes_y_np, py::array_t<float> nodes_z_np,
    py::array_t<float> Jx_np, py::array_t<float> Jy_np, py::array_t<float> Jz_np,
    int32_t id_coil, float j_mag,
    float center_u, float center_v, int axis, bool clear_first)
{
    auto buf_elems = elements_np.request();
    size_t num_elems = buf_elems.shape[0] / 4;

    PhysicsEngine::setup_coil_source(
        static_cast<const int32_t*>(buf_elems.ptr), num_elems,
        static_cast<const int32_t*>(pids_np.request().ptr),
        static_cast<const float*>(nodes_x_np.request().ptr),
        static_cast<const float*>(nodes_y_np.request().ptr),
        static_cast<const float*>(nodes_z_np.request().ptr),
        static_cast<float*>(Jx_np.request().ptr),
        static_cast<float*>(Jy_np.request().ptr),
        static_cast<float*>(Jz_np.request().ptr),
        id_coil, j_mag, center_u, center_v, axis, clear_first);
}

void assemble_magnetostatics(MatrixCSR& K, py::array_t<float> f_np,
    py::array_t<int32_t> elements_np,
    py::array_t<float> inv_mu_np,
    py::array_t<float> Jx_np, py::array_t<float> Jy_np, py::array_t<float> Jz_np,
    float mu0,
    py::array_t<float> nodes_x_np,
    py::array_t<float> nodes_y_np,
    py::array_t<float> nodes_z_np)
{
    auto buf_f = f_np.request();
    Vector f_cpp(static_cast<float*>(buf_f.ptr), buf_f.shape[0]);

    auto buf_elems = elements_np.request();
    size_t num_elems = buf_elems.shape[0] / 4;

    PhysicsEngine::assemble_physics(K, f_cpp,
        static_cast<const int32_t*>(buf_elems.ptr), num_elems,
        static_cast<const float*>(nodes_x_np.request().ptr),
        static_cast<const float*>(nodes_y_np.request().ptr),
        static_cast<const float*>(nodes_z_np.request().ptr),
        static_cast<const float*>(inv_mu_np.request().ptr),
        static_cast<const float*>(Jx_np.request().ptr),
        static_cast<const float*>(Jy_np.request().ptr),
        static_cast<const float*>(Jz_np.request().ptr),
        mu0);
}

void assemble_transient_bridge(
    MatrixCSR& K_eff, MatrixCSR& M, py::array_t<float> f0_np,
    py::array_t<int32_t> elements_np,
    py::array_t<float> inv_mu_np,
    py::array_t<float> sigma_np,
    py::array_t<float> Jx_np, py::array_t<float> Jy_np, py::array_t<float> Jz_np,
    float mu0, float dt,
    py::array_t<float> nodes_x_np,
    py::array_t<float> nodes_y_np,
    py::array_t<float> nodes_z_np)
{
    auto buf_f0 = f0_np.request();
    Vector f0_cpp(static_cast<float*>(buf_f0.ptr), buf_f0.shape[0]);

    auto buf_elems = elements_np.request();
    size_t num_elems = buf_elems.shape[0] / 4;

    PhysicsEngine::assemble_transient_system(K_eff, M, f0_cpp,
        static_cast<const int32_t*>(buf_elems.ptr), num_elems,
        static_cast<const float*>(nodes_x_np.request().ptr),
        static_cast<const float*>(nodes_y_np.request().ptr),
        static_cast<const float*>(nodes_z_np.request().ptr),
        static_cast<const float*>(inv_mu_np.request().ptr),
        static_cast<const float*>(sigma_np.request().ptr),
        static_cast<const float*>(Jx_np.request().ptr),
        static_cast<const float*>(Jy_np.request().ptr),
        static_cast<const float*>(Jz_np.request().ptr),
        mu0, dt);
}

void apply_dirichlet_bc(MatrixCSR& K,
    py::array_t<float> nodes_x_np, py::array_t<float> nodes_y_np, py::array_t<float> nodes_z_np,
    float tol, float penalty)
{
    auto buf_nx = nodes_x_np.request();
    size_t num_nodes = buf_nx.shape[0];

    PhysicsEngine::apply_dirichlet_bounding_box(K,
        static_cast<const float*>(buf_nx.ptr),
        static_cast<const float*>(nodes_y_np.request().ptr),
        static_cast<const float*>(nodes_z_np.request().ptr),
        num_nodes, tol, penalty);
}

void compute_b_field_bridge(
    py::array_t<int32_t> elements_np,
    py::array_t<float> nodes_x_np, py::array_t<float> nodes_y_np, py::array_t<float> nodes_z_np,
    py::array_t<float> A_np,
    py::array_t<float> Bx_np, py::array_t<float> By_np, py::array_t<float> Bz_np, py::array_t<float> Bmag_np)
{
    auto buf_elems = elements_np.request();
    size_t num_elems = buf_elems.shape[0] / 4;

    PhysicsEngine::compute_b_field(
        static_cast<const int32_t*>(buf_elems.ptr), num_elems,
        static_cast<const float*>(nodes_x_np.request().ptr),
        static_cast<const float*>(nodes_y_np.request().ptr),
        static_cast<const float*>(nodes_z_np.request().ptr),
        static_cast<const float*>(A_np.request().ptr),
        static_cast<float*>(Bx_np.request().ptr),
        static_cast<float*>(By_np.request().ptr),
        static_cast<float*>(Bz_np.request().ptr),
        static_cast<float*>(Bmag_np.request().ptr));
}

void compute_eddy_current_bridge(
    py::array_t<int32_t> elements_np,
    py::array_t<float> sigma_np,
    py::array_t<float> A_curr_np,
    py::array_t<float> A_prev_np,
    float dt,
    py::array_t<float> Jx_eddy_np, py::array_t<float> Jy_eddy_np,
    py::array_t<float> Jz_eddy_np, py::array_t<float> Jmag_eddy_np)
{
    auto buf_elems = elements_np.request();
    size_t num_elems = buf_elems.shape[0] / 4;

    PhysicsEngine::compute_eddy_current(
        static_cast<const int32_t*>(buf_elems.ptr), num_elems,
        static_cast<const float*>(sigma_np.request().ptr),
        static_cast<const float*>(A_curr_np.request().ptr),
        static_cast<const float*>(A_prev_np.request().ptr),
        dt,
        static_cast<float*>(Jx_eddy_np.request().ptr),
        static_cast<float*>(Jy_eddy_np.request().ptr),
        static_cast<float*>(Jz_eddy_np.request().ptr),
        static_cast<float*>(Jmag_eddy_np.request().ptr));
}

// -----------------------------------------------------------------------------
// MODULE EXPOSURE TO PYTHON
// -----------------------------------------------------------------------------
PYBIND11_MODULE(fem_simulator, m) {
    m.doc() = "High-Performance FEM Numerical Engine in C++/CUDA";

    py::class_<MatrixCSR>(m, "MatrixCSR")
        .def(py::init<size_t, size_t>(), py::arg("rows"), py::arg("cols"))
        .def("rows", &MatrixCSR::rows)
        .def("cols", &MatrixCSR::cols)
        .def("nnz", &MatrixCSR::nnz)
        .def("add_value", &MatrixCSR::add_value);

    m.def("assemble_topology", &assemble_topology_csr, 
          "Builds the CSR skeleton from NumPy coordinate arrays",
          py::arg("matrix"), py::arg("rows_np"), py::arg("cols_np"));

    m.def("assemble_topology_elements", &assemble_topology_from_elements,
          "Builds the CSR skeleton directly from the flattened tetrahedra array in C++",
          py::arg("matrix"), py::arg("elements_np"));

    m.def("setup_materials", &setup_materials_bridge,
          "Assigns 1/mu_r and optional sigma for any number of (id, mu_r[, sigma]) tuples via *args",
          py::arg("pids"), py::arg("inv_mu"), py::arg("sigma") = py::none(),
          py::arg("default_mu_r") = 1.0f, py::arg("default_sigma") = 0.0f);

    m.def("setup_coil_source", &setup_coil_source_bridge,
          "Assigns azimuthal current density J for a cylindrical coil along axis (0=X, 1=Y, 2=Z)",
          py::arg("elements"), py::arg("pids"),
          py::arg("nodes_x"), py::arg("nodes_y"), py::arg("nodes_z"),
          py::arg("Jx"), py::arg("Jy"), py::arg("Jz"),
          py::arg("id_coil"), py::arg("j_mag"),
          py::arg("center_u"), py::arg("center_v"),
          py::arg("axis") = 0, py::arg("clear_first") = true);

    m.def("assemble_magnetostatics", &assemble_magnetostatics,
          "Ensambla K y f para magnetostática vectorial",
          py::arg("K"), py::arg("f"), py::arg("elements"),
          py::arg("inv_mu"), py::arg("Jx"), py::arg("Jy"), py::arg("Jz"),
          py::arg("mu0"), py::arg("nodes_x"), py::arg("nodes_y"), py::arg("nodes_z"));

    m.def("assemble_transient", &assemble_transient_bridge,
          "Ensambla K_eff = K + M/dt, M y f0 para el régimen cuasiestático transitorio",
          py::arg("K_eff"), py::arg("M"), py::arg("f0"), py::arg("elements"),
          py::arg("inv_mu"), py::arg("sigma"), py::arg("Jx"), py::arg("Jy"), py::arg("Jz"),
          py::arg("mu0"), py::arg("dt"), py::arg("nodes_x"), py::arg("nodes_y"), py::arg("nodes_z"));

    m.def("apply_dirichlet_bc", &apply_dirichlet_bc,
          "Applies Dirichlet penalty on the bounding box nodes in C++",
          py::arg("K"), py::arg("nodes_x"), py::arg("nodes_y"), py::arg("nodes_z"),
          py::arg("tol") = 1e-3f, py::arg("penalty") = 1e20f);

    m.def("solve_cg_gpu", &solve_system_gpu, 
          "Solves Ax=b using Conjugate Gradient on GPU (CUDA)",
          py::arg("A"), py::arg("b"), py::arg("tol") = 1e-6f, py::arg("max_iter") = 1000);

    m.def("solve_cg_cpu", &solve_system_cpu, 
          "Solves Ax=b using Conjugate Gradient on CPU",
          py::arg("A"), py::arg("b"), py::arg("tol") = 1e-6f, py::arg("max_iter") = 1000);

    m.def("solve_transient_eddy_gpu", &solve_transient_eddy_gpu_bridge,
          "Runs the full time-domain eddy current loop in GPU VRAM",
          py::arg("K_eff"), py::arg("M"), py::arg("f0"),
          py::arg("dt"), py::arg("num_steps"), py::arg("freq"),
          py::arg("save_every") = 1, py::arg("tol") = 1e-6f, py::arg("max_iter") = 1000);

    m.def("solve_transient_barkhausen_gpu", &solve_transient_barkhausen_gpu_bridge,
          "Runs the coupled 3D Eddy-Current + Stochastic ABBM Barkhausen solver in GPU VRAM",
          py::arg("K_eff"), py::arg("M"), py::arg("f0"),
          py::arg("elements"), py::arg("pids"), py::arg("target_pid"),
          py::arg("sensor_bounds") = py::none(),
          py::arg("nodes_x"), py::arg("nodes_y"), py::arg("nodes_z"),
          py::arg("inv_mu"), py::arg("sigma"), py::arg("mu0"), py::arg("dt"),
          py::arg("num_steps"), py::arg("freq"), py::arg("save_every") = 10,
          py::arg("num_domains") = 250, py::arg("G_const") = 0.1356f,
          py::arg("h_param") = 1.5e-4f, py::arg("Msat") = 4.5e5f,
          py::arg("mean_hc") = 1200.0f, py::arg("std_hc") = 650.0f,
          py::arg("h_anh") = 2800.0f, py::arg("corr_length") = 0.008f,
          py::arg("avalanche_intensity") = 0.85f, py::arg("alpha") = 0.12f,
          py::arg("mag_axis") = py::make_tuple(1.0f, 0.0f, 0.0f),
          py::arg("seed") = 42, py::arg("tol") = 1e-6f, py::arg("max_iter") = 1000);

    m.def("compute_b_field", &compute_b_field_bridge,
          "Computes B = curl(A) and |B| per element in C++",
          py::arg("elements"), py::arg("nodes_x"), py::arg("nodes_y"), py::arg("nodes_z"),
          py::arg("A"), py::arg("Bx"), py::arg("By"), py::arg("Bz"), py::arg("Bmag"));

    m.def("compute_eddy_current", &compute_eddy_current_bridge,
          "Computes J_eddy = -sigma * dA/dt and |J_eddy| per element in C++",
          py::arg("elements"), py::arg("sigma"), py::arg("A_curr"), py::arg("A_prev"),
          py::arg("dt"), py::arg("Jx_eddy"), py::arg("Jy_eddy"), py::arg("Jz_eddy"), py::arg("Jmag_eddy"));
}