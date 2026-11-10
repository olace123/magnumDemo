import numpy as np
import gmsh
import pyvista as pv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "bin" / "Release"))

import fem_simulator

# --- 1. GENERAL CONFIGURATION ---
msh_path = project_root / "python" / "msh"
mesh_filename = str(msh_path / "BarkhausenFinal1.msh")

# Physical IDs
ID_SAMPLE = 2
ID_CORE   = 3
ID_COIL   = 4
ID_AIR    = 5

# --- 2. ELECTROMAGNETIC PROPERTIES ---
MU_R_AIR   = 1.0
MU_R_COIL = 1.0
MU_R_CORE   = 2000.0
MU_R_SAMPLE = 500.0

# Electrical conductivity sigma (S/m) -> Activates Eddy Currents in Core and Sample
SIGMA_AIR   = 0.0
SIGMA_COIL = 0.0     # Coil carries imposed current J_s(t)
SIGMA_CORE   = 2.0e6   # Iron/Electrical steel (~2 MS/m)
SIGMA_SAMPLE = 5.0e6   # Conductive sample (~5 MS/m)

J_MAGNITUDE = 1e8

# --- 3. TIME DOMAIN PARAMETERS ---
FREQ        = 50.0                  # AC excitation frequency (50 Hz -> period T = 0.02 s)
STEPS_CYCLE = 40                    # 40 steps per cycle
DT          = (1.0 / FREQ) / STEPS_CYCLE  # dt = 5e-4 s
NUM_STEPS   = 80                    # Simulate 1 full cycle
SAVE_EVERY  = 2                     # Save 1 frame every 2 steps (20 frames for ParaView)


def run_transient_simulation(filename):
    print("--- 1. STARTING TRANSIENT QUASISTATIC SOLVER (C++/CUDA) ---")
    gmsh.initialize()
    try:
        gmsh.open(filename)
    except Exception as e:
        print(f"Error opening msh: {e}")
        gmsh.finalize()
        return None

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords, dtype=np.float32).reshape(-1, 3)
    num_nodes = len(nodes)

    node_tags_np = np.asarray(node_tags, dtype=np.int32)
    max_tag = int(np.max(node_tags_np))
    tag_map = np.zeros(max_tag + 1, dtype=np.int32)
    tag_map[node_tags_np] = np.arange(num_nodes, dtype=np.int32)

    nodes_x = np.ascontiguousarray(nodes[:, 0], dtype=np.float32)
    nodes_y = np.ascontiguousarray(nodes[:, 1], dtype=np.float32)
    nodes_z = np.ascontiguousarray(nodes[:, 2], dtype=np.float32)

    elements_list, pids_list, coil_node_indices = [], [], []
    for pid in [ID_SAMPLE, ID_CORE, ID_COIL, ID_AIR]:
        entities = gmsh.model.getEntitiesForPhysicalGroup(3, pid)
        for entity in entities:
            etypes, _, enodes_list = gmsh.model.mesh.getElements(3, entity)
            for i, etype in enumerate(etypes):
                if etype == 4:
                    conn_tags = np.array(enodes_list[i], dtype=np.int32).reshape(-1, 4)
                    elems = tag_map[conn_tags]
                    elements_list.append(elems)
                    pids_list.append(np.full(len(elems), pid, dtype=np.int32))
                    if pid == ID_COIL:
                        coil_node_indices.append(elems.ravel())

    gmsh.finalize()

    elements_2d = np.vstack(elements_list).astype(np.int32)
    elements_np = np.ascontiguousarray(elements_2d.ravel(), dtype=np.int32)
    pids_np = np.concatenate(pids_list).astype(np.int32)
    num_elements = len(elements_2d)

    if coil_node_indices:
        unique_coil_nodes = np.unique(np.concatenate(coil_node_indices))
        coil_center = np.mean(nodes[unique_coil_nodes], axis=0)
    else:
        coil_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    print(f"  → {num_nodes:,} nodes and {num_elements:,} tetrahedrons collected")

    # Build CSR topology for K_eff and M
    print("  Building CSR topology for K_eff and M in C++...")
    num_dofs = num_nodes * 3
    K_eff = fem_simulator.MatrixCSR(num_dofs, num_dofs)
    M     = fem_simulator.MatrixCSR(num_dofs, num_dofs)
    fem_simulator.assemble_topology_elements(K_eff, elements_np)
    fem_simulator.assemble_topology_elements(M, elements_np)

    # Material properties: 1/mu_r, sigma, and base source J0
    inv_mu_per_elem = np.empty(num_elements, dtype=np.float32)
    sigma_per_elem  = np.empty(num_elements, dtype=np.float32)
    Jx_per_elem     = np.empty(num_elements, dtype=np.float32)
    Jy_per_elem     = np.empty(num_elements, dtype=np.float32)
    Jz_per_elem     = np.empty(num_elements, dtype=np.float32)

    fem_simulator.setup_materials(
        pids_np, inv_mu_per_elem, sigma_per_elem,
        (ID_AIR,    MU_R_AIR,    SIGMA_AIR),
        (ID_COIL,   MU_R_COIL,   SIGMA_COIL),
        (ID_CORE,   MU_R_CORE,   SIGMA_CORE),
        (ID_SAMPLE, MU_R_SAMPLE, SIGMA_SAMPLE)
    )

    fem_simulator.setup_coil_source(
        elements_np, pids_np,
        nodes_x, nodes_y, nodes_z,
        Jx_per_elem, Jy_per_elem, Jz_per_elem,
        id_coil=ID_COIL, j_mag=J_MAGNITUDE,
        center_u=float(coil_center[1]), center_v=float(coil_center[2]),
        axis=0, clear_first=True
    )

    # Assemble transient system: K_eff = K + M/dt, M, and f0
    print("  Assembling K_eff, Mass Matrix M, and f0 vector in C++...")
    mu0 = float(4.0 * np.pi * 1e-7)
    f0_np = np.zeros(num_dofs, dtype=np.float32)

    fem_simulator.assemble_transient(
        K_eff, M, f0_np, elements_np,
        inv_mu_per_elem, sigma_per_elem,
        Jx_per_elem, Jy_per_elem, Jz_per_elem,
        mu0, float(DT), nodes_x, nodes_y, nodes_z
    )

    # Dirichlet boundary conditions on K_eff
    fem_simulator.apply_dirichlet_bc(K_eff, nodes_x, nodes_y, nodes_z, tol=1e-3, penalty=1e20)

    # Solve all time steps on GPU
    print(f"  Executing {NUM_STEPS} time steps on GPU (dt={DT:.2e}s, f={FREQ}Hz)...")
    A_snapshots = fem_simulator.solve_transient_eddy_gpu(
        K_eff, M, f0_np,
        dt=float(DT), num_steps=NUM_STEPS, freq=float(FREQ),
        save_every=SAVE_EVERY, tol=1e-6, max_iter=1000
    )

    J0_vectors = np.column_stack((Jx_per_elem, Jy_per_elem, Jz_per_elem))
    return (nodes, nodes_x, nodes_y, nodes_z,
            elements_2d, elements_np, pids_np, sigma_per_elem,
            J0_vectors, A_snapshots)


# =============================================================================
# TIME SERIES EXPORT (.vtu + .pvd) FOR PARAVIEW ANIMATION
# =============================================================================
res = run_transient_simulation(mesh_filename)
if res is None:
    exit()

(points, nodes_x, nodes_y, nodes_z,
 cells_3d_np, elements_np, vol_ids_3d_np, sigma_per_elem,
 J0_elem, A_snapshots) = res

print("\n--- 2. EXPORTING TIME SERIES FOR PARAVIEW (.pvd) ---")
out_dir = project_root / "python" / "output" / "eddy_results"
out_dir.mkdir(parents=True, exist_ok=True)

num_cells = len(cells_3d_np)
pv_cells_3d = np.hstack((np.full((num_cells, 1), 4, dtype=np.int32), cells_3d_np)).ravel()
types_3d = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

Bx = np.empty(num_cells, dtype=np.float32)
By = np.empty(num_cells, dtype=np.float32)
Bz = np.empty(num_cells, dtype=np.float32)
Bmag = np.empty(num_cells, dtype=np.float32)

Jx_eddy = np.empty(num_cells, dtype=np.float32)
Jy_eddy = np.empty(num_cells, dtype=np.float32)
Jz_eddy = np.empty(num_cells, dtype=np.float32)
Jmag_eddy = np.empty(num_cells, dtype=np.float32)

dt_saved = DT * SAVE_EVERY
num_frames = A_snapshots.shape[0]
pvd_entries = []

for k in range(num_frames):
    t_val = k * dt_saved
    A_curr = np.ascontiguousarray(A_snapshots[k], dtype=np.float32)
    A_prev = np.ascontiguousarray(A_snapshots[k - 1] if k > 0 else A_snapshots[0], dtype=np.float32)

    # 1. Calculate B(t) = curl(A) in C++
    fem_simulator.compute_b_field(elements_np, nodes_x, nodes_y, nodes_z, A_curr, Bx, By, Bz, Bmag)

    # 2. Calculate J_eddy(t) = -sigma * (A_curr - A_prev) / dt_saved in C++
    fem_simulator.compute_eddy_current(
        elements_np, sigma_per_elem, A_curr, A_prev, float(dt_saved),
        Jx_eddy, Jy_eddy, Jz_eddy, Jmag_eddy
    )

    grid_vol = pv.UnstructuredGrid(pv_cells_3d, types_3d, points)
    grid_vol.cell_data['PhysicalID']   = vol_ids_3d_np
    grid_vol.cell_data['J_Source']     = J0_elem * float(np.sin(2.0 * np.pi * FREQ * t_val))
    grid_vol.cell_data['J_Eddy_Elem']  = np.column_stack((Jx_eddy, Jy_eddy, Jz_eddy))
    grid_vol.cell_data['J_Eddy_Mag']   = Jmag_eddy.copy()
    grid_vol.cell_data['B_Field_Elem'] = np.column_stack((Bx, By, Bz))
    grid_vol.cell_data['B_Mag_Elem']   = Bmag.copy()

    grid_vol.point_data['A'] = A_curr.reshape(-1, 3)
    grid_vol = grid_vol.cell_data_to_point_data(pass_cell_data=True)
    grid_vol.point_data['B_Field'] = grid_vol.point_data['B_Field_Elem']
    grid_vol.point_data['B_Mag']   = grid_vol.point_data['B_Mag_Elem']
    grid_vol.point_data['J_Eddy']  = grid_vol.point_data['J_Eddy_Elem']

    vtu_name = f"frame_{k:04d}.vtu"
    grid_vol.save(str(out_dir / vtu_name))
    pvd_entries.append((t_val, f"eddy_results/{vtu_name}"))

pvd_path = project_root / "python" / "output" / "eddy_animation.pvd"
with open(pvd_path, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <Collection>\n')
    for t_val, rel_path in pvd_entries:
        f.write(f'    <DataSet timestep="{t_val:.6e}" group="" part="0" file="{rel_path}"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>\n')

print(f"Done! Open '{pvd_path.name}' in ParaView and press Play to see the animation.")