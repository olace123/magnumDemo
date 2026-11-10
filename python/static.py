import numpy as np
import gmsh
import pyvista as pv

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "bin" / "Release"))

import fem_simulator  # compiled C++ module (fem_simulator.so)

# --- 1. GENERAL CONFIGURATION ---
msh_path = project_root / "python" / "msh"
mesh_filename = str(msh_path / "BarkhausenFinal1.msh")

# Physical IDs
ID_SAMPLE = 2
ID_CORE = 3
ID_COIL = 4
ID_AIR = 5

# --- 2. PROPERTIES ---
MU_R_AIR = 1.0
MU_R_COIL = 1.0
MU_R_CORE = 2000.0
MU_R_SAMPLE = 500.0
J_MAGNITUDE = 1e8


# =============================================================================
# PART A: FEM SOLVER (100% IN C++)
# =============================================================================
def run_fem_simulation(filename):
    print("--- 1. STARTING FEM SOLVER (C++) ---")
    gmsh.initialize()
    try:
        gmsh.open(filename)
    except Exception as e:
        print(f"Error: Could not open the msh file ({e}).")
        gmsh.finalize()
        return None

    # 1. Extract nodes (SoA in float32 for C++)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords, dtype=np.float32).reshape(-1, 3)
    num_nodes = len(nodes)

    # Ultra-fast vectorized mapping from Gmsh tag to 0-based index
    max_tag = int(np.max(node_tags))
    tag_map = np.zeros(max_tag + 1, dtype=np.int32)
    tag_map[node_tags.astype(np.int32)] = np.arange(num_nodes, dtype=np.int32)

    nodes_x = np.ascontiguousarray(nodes[:, 0], dtype=np.float32)
    nodes_y = np.ascontiguousarray(nodes[:, 1], dtype=np.float32)
    nodes_z = np.ascontiguousarray(nodes[:, 2], dtype=np.float32)

    # 2. Collect ALL tetrahedrons + physical IDs in a single pass
    print("  Collecting elements from Gmsh...")
    elements_list = []
    pids_list = []
    coil_node_indices = []

    for pid in [ID_SAMPLE, ID_CORE, ID_COIL, ID_AIR]:
        entities = gmsh.model.getEntitiesForPhysicalGroup(3, pid)
        for entity in entities:
            etypes, _, enodes_list = gmsh.model.mesh.getElements(3, entity)
            for i, etype in enumerate(etypes):
                if etype == 4:  # 4-node Tetrahedron
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

    # Coil centroid
    if coil_node_indices:
        unique_coil_nodes = np.unique(np.concatenate(coil_node_indices))
        coil_center = np.mean(nodes[unique_coil_nodes], axis=0)
    else:
        coil_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    print(f"  → {num_nodes:,} nodes and {num_elements:,} tetrahedrons collected")

    # 3. Build CSR topology directly in C++
    print("  Building CSR topology in C++...")
    num_dofs = num_nodes * 3
    K = fem_simulator.MatrixCSR(num_dofs, num_dofs)
    fem_simulator.assemble_topology_elements(K, elements_np)
    print(f"  → Topology built: {K.nnz():,} nnz")

    # 4. Assign material properties (1/mu_r) with *args and J source in C++
    print("  Calculating materials and J current in C++...")
    inv_mu_per_elem = np.empty(num_elements, dtype=np.float32)
    Jx_per_elem = np.empty(num_elements, dtype=np.float32)
    Jy_per_elem = np.empty(num_elements, dtype=np.float32)
    Jz_per_elem = np.empty(num_elements, dtype=np.float32)

    fem_simulator.setup_materials(
        pids_np, inv_mu_per_elem, None,
        (ID_AIR,    MU_R_AIR),
        (ID_COIL,   MU_R_COIL),
        (ID_CORE,   MU_R_CORE),
        (ID_SAMPLE, MU_R_SAMPLE)
    )

    fem_simulator.setup_coil_source(
        elements_np, pids_np,
        nodes_x, nodes_y, nodes_z,
        Jx_per_elem, Jy_per_elem, Jz_per_elem,
        id_coil=ID_COIL, j_mag=J_MAGNITUDE,
        center_u=float(coil_center[1]), center_v=float(coil_center[2]),
        axis=0, clear_first=True
    )

    # 5. Assembly of stiffness matrix K and force vector f in C++
    print("  Assembling K and f in C++...")
    mu0 = float(4.0 * np.pi * 1e-7)
    f_np = np.zeros(num_dofs, dtype=np.float32)

    fem_simulator.assemble_magnetostatics(
        K, f_np, elements_np,
        inv_mu_per_elem, Jx_per_elem, Jy_per_elem, Jz_per_elem,
        mu0, nodes_x, nodes_y, nodes_z
    )

    # 6. Dirichlet boundary conditions (in C++)
    print("  Applying Dirichlet BCs in C++...")
    fem_simulator.apply_dirichlet_bc(K, nodes_x, nodes_y, nodes_z, tol=1e-3, penalty=1e20)

    # 7. Solve the Ax = b system with Conjugate Gradient (in C++)
    print("  Solving Ax=b system with CG (C++)...")
    A_full = fem_simulator.solve_cg_gpu(K, f_np)

    # 8. Post-processing: Compute B = curl(A) and |B| in C++
    print("  Computing magnetic field B = curl(A) in C++...")
    Bx_elem = np.empty(num_elements, dtype=np.float32)
    By_elem = np.empty(num_elements, dtype=np.float32)
    Bz_elem = np.empty(num_elements, dtype=np.float32)
    Bmag_elem = np.empty(num_elements, dtype=np.float32)

    fem_simulator.compute_b_field(
        elements_np, nodes_x, nodes_y, nodes_z,
        A_full, Bx_elem, By_elem, Bz_elem, Bmag_elem
    )

    J_vectors = np.column_stack((Jx_per_elem, Jy_per_elem, Jz_per_elem))
    B_vectors_elem = np.column_stack((Bx_elem, By_elem, Bz_elem))

    return nodes, elements_2d, pids_np, A_full.reshape(-1, 3), B_vectors_elem, Bmag_elem, J_vectors


# =============================================================================
# PART B: DIRECT EXPORT TO PARAVIEW (.vtu)
# =============================================================================
result = run_fem_simulation(mesh_filename)
if result is None:
    exit()

points, cells_3d_np, vol_ids_3d_np, A_vectors, B_elem, Bmag_elem, J_elem = result

print("\n--- 2. EXPORTING TO PARAVIEW (.vtu) ---")
num_cells = len(cells_3d_np)

pv_cells_3d = np.hstack((np.full((num_cells, 1), 4, dtype=np.int32), cells_3d_np)).ravel()
types_3d = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

grid_vol = pv.UnstructuredGrid(pv_cells_3d, types_3d, points)

grid_vol.cell_data['PhysicalID'] = vol_ids_3d_np
grid_vol.cell_data['J_Field'] = J_elem
grid_vol.cell_data['B_Field_Elem'] = B_elem
grid_vol.cell_data['B_Mag_Elem'] = Bmag_elem

grid_vol.point_data['A'] = A_vectors
grid_vol = grid_vol.cell_data_to_point_data(pass_cell_data=True)
grid_vol.point_data['B_Field'] = grid_vol.point_data['B_Field_Elem']
grid_vol.point_data['B_Mag'] = grid_vol.point_data['B_Mag_Elem']

out_dir = project_root / "python" / "output"
out_dir.mkdir(parents=True, exist_ok=True)
output_filename = out_dir / "static.vtu"
grid_vol.save(str(output_filename))
print(f"Done. File '{output_filename}' generated successfully.")