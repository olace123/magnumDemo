import numpy as np
import gmsh
import matplotlib.pyplot as plt
from scipy.special import j0, j1, jn_zeros
import time
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "bin" / "Release"))

import fem_simulator

# =============================================================================
# 1. CANONICAL MAGNETIC DIFFUSION BENCHMARK CONFIGURATION (h-FEM)
# =============================================================================
msh_dir = project_root / "python" / "msh"
msh_dir.mkdir(parents=True, exist_ok=True)

R_CORE = 0.020   # Conductor cylinder radius: 20 mm
R_COIL = 0.028   # Outer coil radius: 28 mm
L_BOX  = 0.090   # Outer box: 90 mm
H_CYL  = 0.040   # Periodic height/3D cross-section: 40 mm (no air gap top/bottom)
H_BOX  = H_CYL   # Cylinder goes from cap to cap (z = -H/2 to +H/2) -> eliminates 3D axial leakage (d/dz = 0)

ID_CORE   = 3
ID_COIL   = 4
ID_AIR    = 5

# Physical parameters: Skin depth δ = 11.25 mm (R_CORE / δ = 1.78)
MU0         = float(4.0 * np.pi * 1e-7)
MU_R_CORE   = 1.0
SIGMA_CORE  = 4.0e7    # 40 MS/m -> δ = 11.25 mm
J_MAGNITUDE = 1.0e7    # 10 MA/m^2

FREQ        = 50.0     # 50 Hz (T = 20 ms)
STEPS_CYCLE = 80       # 80 steps per cycle (dt = 0.25 ms)
DT          = (1.0 / FREQ) / STEPS_CYCLE
NUM_CYCLES  = 2        # 2 full cycles (160 steps)
NUM_STEPS   = STEPS_CYCLE * NUM_CYCLES
SAVE_EVERY  = 1

DELTA_SKIN = 1.0 / np.sqrt(np.pi * FREQ * MU0 * MU_R_CORE * SIGMA_CORE)

# Target mesh sizes in the core (in meters) for h-refinement study
MESH_SIZES_CORE = [0.0075, 0.0050, 0.0034, 0.0023]


# =============================================================================
# 2. PARAMETRIC 3D MESH GENERATOR WITH GMSH (EQUIVALENT INFINITE CYLINDER)
# =============================================================================
def generate_parametric_mesh(lc_core, filename_msh):
    if filename_msh.exists():
        return
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("benchmark_kelvin_bessel")

    box_tag  = gmsh.model.occ.addBox(-L_BOX/2, -L_BOX/2, -H_BOX/2, L_BOX, L_BOX, H_BOX)
    coil_tag = gmsh.model.occ.addCylinder(0, 0, -H_CYL/2, 0, 0, H_CYL, R_COIL)
    core_tag = gmsh.model.occ.addCylinder(0, 0, -H_CYL/2, 0, 0, H_CYL, R_CORE)

    gmsh.model.occ.fragment([(3, box_tag)], [(3, coil_tag), (3, core_tag)])
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    core_vols, coil_vols, air_vols = [], [], []
    for dim, tag in vols:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        if xmax <= R_CORE + 1e-4:
            core_vols.append(tag)
        elif xmax <= R_COIL + 1e-4:
            coil_vols.append(tag)
        else:
            air_vols.append(tag)

    gmsh.model.addPhysicalGroup(3, core_vols, ID_CORE)
    gmsh.model.addPhysicalGroup(3, coil_vols, ID_COIL)
    gmsh.model.addPhysicalGroup(3, air_vols,  ID_AIR)

    lc_coil = lc_core * 1.25
    lc_air  = lc_core * 2.5

    for v in air_vols:
        pts = gmsh.model.getBoundary([(3, v)], combined=False, oriented=False, recursive=True)
        gmsh.model.mesh.setSize(pts, lc_air)
    for v in coil_vols:
        pts = gmsh.model.getBoundary([(3, v)], combined=False, oriented=False, recursive=True)
        gmsh.model.mesh.setSize(pts, lc_coil)
    for v in core_vols:
        pts = gmsh.model.getBoundary([(3, v)], combined=False, oriented=False, recursive=True)
        gmsh.model.mesh.setSize(pts, lc_core)

    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(filename_msh))
    gmsh.finalize()


# =============================================================================
# 3. EXACT ANALYTICAL SOLUTION (DUHAMEL-FOURIER-BESSEL SERIES FROM t=0)
# =============================================================================
def exact_analytical_solution(r_points, t_array, B_ref_time_series, r_ref, mu, sigma, freq):
    N_modes = 40
    alphas = jn_zeros(0, N_modes)
    dt_val = t_array[1] - t_array[0]
    dB_ref_dt = np.gradient(B_ref_time_series, dt_val)

    Bz_analytic = np.outer(B_ref_time_series, np.ones_like(r_points))
    for n in range(N_modes):
        a_n = alphas[n]
        lam_n = (a_n / r_ref) ** 2 / (mu * sigma)
        spatial_mode = j0(a_n * (r_points / r_ref)) / (a_n * j1(a_n))

        I_n = np.zeros(len(t_array))
        exp_factor = np.exp(-lam_n * dt_val)
        for step in range(1, len(t_array)):
            I_n[step] = I_n[step - 1] * exp_factor + 0.5 * (dB_ref_dt[step] + dB_ref_dt[step - 1]) * ((1.0 - exp_factor) / lam_n)

        Bz_analytic -= 2.0 * np.outer(I_n, spatial_mode)

    return Bz_analytic


# =============================================================================
# 4. RUN SIMULATION AND MEASURE L2 ERROR IN VOLUME-WEIGHTED NORM
# =============================================================================
def run_simulation_and_metrics(lc_target):
    msh_path = msh_dir / f"KelvinBessel_h_{int(lc_target*1e4):03d}.msh"
    generate_parametric_mesh(lc_target, msh_path)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(msh_path))

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords, dtype=np.float32).reshape(-1, 3)
    num_nodes = len(nodes)

    node_tags_np = np.asarray(node_tags, dtype=np.int32)
    tag_map = np.zeros(int(np.max(node_tags_np)) + 1, dtype=np.int32)
    tag_map[node_tags_np] = np.arange(num_nodes, dtype=np.int32)

    nodes_x = np.ascontiguousarray(nodes[:, 0], dtype=np.float32)
    nodes_y = np.ascontiguousarray(nodes[:, 1], dtype=np.float32)
    nodes_z = np.ascontiguousarray(nodes[:, 2], dtype=np.float32)

    elements_list, pids_list = [], []
    for pid in [ID_CORE, ID_COIL, ID_AIR]:
        entities = gmsh.model.getEntitiesForPhysicalGroup(3, pid)
        for entity in entities:
            etypes, _, enodes_list = gmsh.model.mesh.getElements(3, entity)
            for i, etype in enumerate(etypes):
                if etype == 4:
                    conn_tags = np.array(enodes_list[i], dtype=np.int32).reshape(-1, 4)
                    elems = tag_map[conn_tags]
                    elements_list.append(elems)
                    pids_list.append(np.full(len(elems), pid, dtype=np.int32))
    gmsh.finalize()

    elements_2d = np.vstack(elements_list).astype(np.int32)
    elements_np = np.ascontiguousarray(elements_2d.ravel(), dtype=np.int32)
    pids_np     = np.concatenate(pids_list).astype(np.int32)
    num_elements = len(elements_2d)
    num_dofs     = num_nodes * 3

    mask_core = (pids_np == ID_CORE)
    core_elems = elements_2d[mask_core]
    v0 = nodes[core_elems[:, 0]]
    v1 = nodes[core_elems[:, 1]] - v0
    v2 = nodes[core_elems[:, 2]] - v0
    v3 = nodes[core_elems[:, 3]] - v0
    vols_core = np.abs(np.sum(v1 * np.cross(v2, v3), axis=1)) / 6.0

    h_elems_core = (6.0 * np.sqrt(2.0) * vols_core) ** (1.0 / 3.0)
    h_mean_mm = float(np.mean(h_elems_core) * 1e3)

    t0 = time.perf_counter()
    K_eff = fem_simulator.MatrixCSR(num_dofs, num_dofs)
    M     = fem_simulator.MatrixCSR(num_dofs, num_dofs)
    fem_simulator.assemble_topology_elements(K_eff, elements_np)
    fem_simulator.assemble_topology_elements(M, elements_np)

    inv_mu_per_elem = np.empty(num_elements, dtype=np.float32)
    sigma_per_elem  = np.empty(num_elements, dtype=np.float32)
    Jx_per_elem     = np.empty(num_elements, dtype=np.float32)
    Jy_per_elem     = np.empty(num_elements, dtype=np.float32)
    Jz_per_elem     = np.empty(num_elements, dtype=np.float32)

    # 1. Assign the 3 benchmark materials using *args
    fem_simulator.setup_materials(
        pids_np, inv_mu_per_elem, sigma_per_elem,
        (ID_AIR,    1.0,       0.0),
        (ID_COIL,   1.0,       0.0),
        (ID_CORE,   MU_R_CORE, SIGMA_CORE)
    )

    # 2. Assign cylindrical coil directly on the Z axis (axis=2, center X=0, Y=0)
    fem_simulator.setup_coil_source(
        elements_np, pids_np,
        nodes_x, nodes_y, nodes_z,
        Jx_per_elem, Jy_per_elem, Jz_per_elem,
        id_coil=ID_COIL, j_mag=J_MAGNITUDE,
        center_u=0.0, center_v=0.0,
        axis=2, clear_first=True
    )

    f0_np = np.zeros(num_dofs, dtype=np.float32)
    fem_simulator.assemble_transient(
        K_eff, M, f0_np, elements_np,
        inv_mu_per_elem, sigma_per_elem,
        Jx_per_elem, Jy_per_elem, Jz_per_elem,
        MU0, float(DT), nodes_x, nodes_y, nodes_z
    )

    # Infinite cylinder boundary condition
    tol = 1e-4
    penalty = 1e12
    min_x, max_x = np.min(nodes_x), np.max(nodes_x)
    min_y, max_y = np.min(nodes_y), np.max(nodes_y)
    min_z, max_z = np.min(nodes_z), np.max(nodes_z)

    for i in range(num_nodes):
        on_side = (abs(nodes_x[i] - min_x) < tol or abs(nodes_x[i] - max_x) < tol or
                   abs(nodes_y[i] - min_y) < tol or abs(nodes_y[i] - max_y) < tol)
        on_cap  = (abs(nodes_z[i] - min_z) < tol or abs(nodes_z[i] - max_z) < tol)
        if on_side:
            K_eff.add_value(3*i + 0, 3*i + 0, penalty)
            K_eff.add_value(3*i + 1, 3*i + 1, penalty)
            K_eff.add_value(3*i + 2, 3*i + 2, penalty)
        elif on_cap:
            K_eff.add_value(3*i + 2, 3*i + 2, penalty)

    A_snapshots = fem_simulator.solve_transient_eddy_gpu(
        K_eff, M, f0_np,
        dt=float(DT), num_steps=NUM_STEPS, freq=float(FREQ),
        save_every=SAVE_EVERY, tol=1e-6, max_iter=1500
    )
    gpu_time = time.perf_counter() - t0

    centroids_core = np.mean(nodes[core_elems], axis=1)
    r_core = np.sqrt(centroids_core[:, 0]**2 + centroids_core[:, 1]**2)

    r_edge_band = (r_core > R_CORE - 0.0025) & (r_core <= R_CORE)
    r_ref_eff   = float(np.average(r_core[r_edge_band], weights=vols_core[r_edge_band]))

    mask_eval   = r_core <= r_ref_eff
    r_mid       = r_core[mask_eval]
    vols_mid    = vols_core[mask_eval]
    mask_center = r_mid < 0.0035

    Bx = np.empty(num_elements, dtype=np.float32)
    By = np.empty(num_elements, dtype=np.float32)
    Bz = np.empty(num_elements, dtype=np.float32)
    Bmag = np.empty(num_elements, dtype=np.float32)

    num_frames = A_snapshots.shape[0]
    t_axis = np.arange(num_frames) * DT * SAVE_EVERY
    Bz_core_all = np.zeros((num_frames, np.sum(mask_core)))

    for k in range(num_frames):
        A_curr = np.ascontiguousarray(A_snapshots[k], dtype=np.float32)
        fem_simulator.compute_b_field(elements_np, nodes_x, nodes_y, nodes_z, A_curr, Bx, By, Bz, Bmag)
        Bz_core_all[k, :] = Bz[mask_core]

    Bz_num_edge   = np.average(Bz_core_all[:, r_edge_band], weights=vols_core[r_edge_band], axis=1)
    Bz_mid_all    = Bz_core_all[:, mask_eval]
    Bz_num_center = np.average(Bz_mid_all[:, mask_center], weights=vols_mid[mask_center], axis=1)

    Bz_ana_all = exact_analytical_solution(
        r_mid, t_axis, Bz_num_edge, r_ref_eff, MU0 * MU_R_CORE, SIGMA_CORE, FREQ
    )

    s0 = STEPS_CYCLE
    diff_sq = (Bz_mid_all[s0:, :] - Bz_ana_all[s0:, :]) ** 2
    ref_sq  = (Bz_ana_all[s0:, :]) ** 2

    num_integral = np.sum(diff_sq * vols_mid[None, :]) * DT
    den_integral = np.sum(ref_sq  * vols_mid[None, :]) * DT
    err_L2_domain = float(np.sqrt(num_integral / den_integral) * 100.0)

    Bz_ana_center = np.average(Bz_ana_all[:, mask_center], weights=vols_mid[mask_center], axis=1)
    err_L2_center = float(
        np.linalg.norm(Bz_num_center[s0:] - Bz_ana_center[s0:]) / np.linalg.norm(Bz_ana_center[s0:]) * 100.0
    )

    return {
        "lc_target_mm": lc_target * 1e3,
        "h_mean_mm": h_mean_mm,
        "h_over_delta": (h_mean_mm * 1e-3) / DELTA_SKIN,
        "num_nodes": num_nodes,
        "num_dofs": num_dofs,
        "num_elems": num_elements,
        "num_core_elems": int(np.sum(mask_core)),
        "err_L2_domain": err_L2_domain,
        "err_L2_center": err_L2_center,
        "gpu_time": gpu_time,
        "t_axis": t_axis,
        "r_mid": r_mid,
        "Bz_mid_all": Bz_mid_all,
        "Bz_ana_all": Bz_ana_all,
        "Bz_num_center": Bz_num_center,
        "Bz_ana_center": Bz_ana_center,
        "Bz_num_edge": Bz_num_edge,
    }


# =============================================================================
# 5. EXECUTE STUDY AND GENERATE FORMAL THESIS REPORT
# =============================================================================
print("================================================================================")
print(f" FORMAL h-FEM CONVERGENCE STUDY (Skin Depth δ = {DELTA_SKIN*1e3:.2f} mm)")
print("================================================================================")

results = []
for lc in MESH_SIZES_CORE:
    print(f"\n-> Evaluating mesh with nominal size h_core = {lc*1e3:.2f} mm...")
    res = run_simulation_and_metrics(lc)
    results.append(res)
    print(f"   Nodes: {res['num_nodes']:,} | Total Tets: {res['num_elems']:,} | "
          f"h_real: {res['h_mean_mm']:.2f} mm | Global L2 Error: {res['err_L2_domain']:.2f}% | "
          f"Center L2 Error: {res['err_L2_center']:.2f}% | GPU Time: {res['gpu_time']:.2f}s")

h_vals  = np.array([r["h_mean_mm"] for r in results])
err_dom = np.array([r["err_L2_domain"] for r in results])
err_cen = np.array([r["err_L2_center"] for r in results])

p_domain, _ = np.polyfit(np.log(h_vals), np.log(err_dom), 1)
p_center, _ = np.polyfit(np.log(h_vals), np.log(err_cen), 1)

print("\n==========================================================================================")
print("                  h-FEM SPATIAL CONVERGENCE TABLE (THESIS SUMMARY)")
print("==========================================================================================")
print(f"{'Mesh':<7} | {'h_core (mm)':<11} | {'h / δ':<7} | {'DOFs':<9} | {'Core Tets':<11} | {'L2 Err Dom (%)':<14} | {'L2 Err Axis (%)':<14} | {'Order p':<7}")
print("-" * 98)
for i, r in enumerate(results):
    if i == 0:
        p_local_str = "  ---  "
    else:
        p_loc = np.log(results[i-1]["err_L2_domain"] / r["err_L2_domain"]) / np.log(results[i-1]["h_mean_mm"] / r["h_mean_mm"])
        p_local_str = f"{p_loc:6.2f}"
    print(f"Mesh {i+1:<1} | {r['h_mean_mm']:11.2f} | {r['h_over_delta']:7.3f} | {r['num_dofs']:9,d} | "
          f"{r['num_core_elems']:11,d} | {r['err_L2_domain']:13.2f}% | {r['err_L2_center']:13.2f}% | {p_local_str}")
print("-" * 98)
print(f" -> Global convergence rate (Log-Log Regression): p = {p_domain:.2f} (Axis Order: p = {p_center:.2f})")
print("==========================================================================================\n")

# LaTeX code ready to paste into thesis
print("% --- LATEX CODE FOR YOUR MASTER'S THESIS ---")
print("\\begin{table}[htbp]")
print("\\centering")
print(f"\\caption{{Spatial $h$-FEM convergence study of the GPU transient solver ($f = 50$ Hz, $\\delta = {DELTA_SKIN*1e3:.2f}$ mm).}}")
print("\\begin{tabular}{cccccccc}")
print("\\hline")
print("Mesh & $h_{\\text{mean}}$ (mm) & $h/\\delta$ & DOFs & Tets ($\\Omega_c$) & $L^2(\\Omega_c)$ Error (\\%) & $L^2(r\\approx 0)$ Error (\\%) & Order $p$ \\\\")
print("\\hline")
for i, r in enumerate(results):
    p_str = "---" if i == 0 else f"{np.log(results[i-1]['err_L2_domain']/r['err_L2_domain'])/np.log(results[i-1]['h_mean_mm']/r['h_mean_mm']):.2f}"
    print(f"M{i+1} & {r['h_mean_mm']:.2f} & {r['h_over_delta']:.3f} & {r['num_dofs']:,} & {r['num_core_elems']:,} & {r['err_L2_domain']:.2f}\\% & {r['err_L2_center']:.2f}\\% & {p_str} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}\n")

# =============================================================================
# 6. VALIDATION AND CONVERGENCE PLOTS PANEL FOR THESIS
# =============================================================================
finest = results[-1]
t_ms = finest["t_axis"] * 1e3

fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5))

h_fit = np.linspace(min(h_vals)*0.9, max(h_vals)*1.1, 50)
err_ref_p1 = err_dom[-1] * (h_fit / h_vals[-1]) ** 1.0

axes[0].loglog(h_vals, err_dom, 'bo-', lw=2.2, ms=7, label=f"Global Error $L^2(\\Omega_c)$ ($p={p_domain:.2f}$)")
axes[0].loglog(h_vals, err_cen, 'rs--', lw=1.8, ms=6, label=f"Central Axis Error $r\\approx 0$ ($p={p_center:.2f}$)")
axes[0].loglog(h_fit, err_ref_p1, 'k:', lw=1.8, label="Theoretical Slope $P_1$ Elements: $\\mathcal{O}(h^1)$")
axes[0].set_title("1. Spatial Convergence $h$-FEM (Log-Log Scale)")
axes[0].set_xlabel("Characteristic element size $h$ (mm)")
axes[0].set_ylabel("Relative $L^2$ Error (%)")
axes[0].grid(True, which="both", ls="--", alpha=0.4)
axes[0].legend(fontsize=9)

s0 = STEPS_CYCLE
axes[1].plot(t_ms[s0:], finest["Bz_num_edge"][s0:], 'b-', lw=2.0, label="Outer Edge $r \\approx R$ (3D FEM)")
axes[1].plot(t_ms[s0:], finest["Bz_ana_center"][s0:], 'k-', lw=2.2, label="Analytical Kelvin-Bessel in Center ($r \\approx 0$)")
axes[1].plot(t_ms[s0:], finest["Bz_num_center"][s0:], 'ro', ms=4.5, alpha=0.8, label=f"CUDA FEM in Center (Err={finest['err_L2_center']:.2f}%)")
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].set_title(f"2. Steady-State Temporal Response ($h={finest['h_mean_mm']:.2f}$ mm)")
axes[1].set_xlabel("Time $t$ (ms)")
axes[1].set_ylabel("Axial Magnetic Field $B_z$ (T)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9)

r_mm = finest["r_mid"] * 1e3
sort_idx = np.argsort(r_mm)
amp_num_per_elem = np.max(np.abs(finest["Bz_mid_all"][s0:, :]), axis=0)
amp_ana_per_elem = np.max(np.abs(finest["Bz_ana_all"][s0:, :]), axis=0)

axes[2].plot(r_mm[sort_idx], amp_ana_per_elem[sort_idx], 'k-', lw=2.5, label="Analytical Envelope $|B_z(r)|$ (Kelvin-Bessel)")
axes[2].scatter(r_mm, amp_num_per_elem, c='crimson', s=14, alpha=0.5, label="Amplitude in 3D FEM Tetrahedrons (CUDA)")
axes[2].set_title(f"3. Radial Amplitude Profile $|B_z(r)|$ (Skin Effect $\\delta={DELTA_SKIN*1e3:.1f}$ mm)")
axes[2].set_xlabel("Radius $r$ (mm)")
axes[2].set_ylabel("Peak Amplitude $|B_z|$ (T)")
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=9)

plt.tight_layout()
out_dir = project_root / "python" / "output"
out_dir.mkdir(parents=True, exist_ok=True)
out_img = out_dir / "h_fem_convergence_thesis.png"
plt.savefig(str(out_img), dpi=220)
print(f"Thesis plot successfully saved at: {out_img}")