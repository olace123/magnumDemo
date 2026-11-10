import numpy as np
import gmsh
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "bin" / "Release"))

import fem_simulator

# --- 1. GEOMETRIC CONFIGURATION (BarkhausenFinal1.msh) ---
msh_path = project_root / "python" / "msh"
mesh_filename = str(msh_path / "BarkhausenFinal1.msh")

ID_SAMPLE = 2   # Plate where the Barkhausen model is applied
ID_CORE   = 3   # U-shaped ferromagnetic yoke
ID_COIL   = 4   # Upper exciter coil
ID_AIR    = 5   # Air box

# Virtual MBN Sensor Region:
SENSOR_BOUNDS = (-0.0012, 0.0012, -0.0012, 0.0012, 0.0008, 0.0015)

# --- 2. REALISTIC ELECTROMAGNETIC PROPERTIES (FERROMAGNETIC STEEL) ---
MU0         = float(4.0 * np.pi * 1e-7)
MU_R_AIR    = 1.0
MU_R_COIL   = 1.0
MU_R_CORE   = 2000.0
MU_R_SAMPLE = 150.0

SIGMA_AIR    = 0.0
SIGMA_COIL   = 0.0
SIGMA_CORE   = 1.5e6
SIGMA_SAMPLE = 3.5e6

J_MAGNITUDE = 8.0e6

# --- 3. BERTOTTI STOCHASTIC MODEL (ABBM) PHYSICAL PARAMETERS ---
NUM_DOMAINS         = 250               # Micro-domains per tetrahedron
G_CONST             = 0.1356            # Bertotti's universal damping constant
H_PARAM             = 3.5e-6            # Fast wall damping scale (tau_m ~ 0.25 ms)
MSAT                = 4.5e5             # Saturation irreversible magnetization (A/m)
MEAN_HC             = 1150.0            # Mean pinning coercive field <Hc> (A/m)
STD_HC              = 700.0             # Microstructural defect dispersion (A/m)
H_ANH               = 3200.0            # Anhysteretic restoration characteristic field H_a (A/m)
CORR_LENGTH         = 0.0055            # Ornstein-Uhlenbeck repinning correlation length (xi_m / Msat)
AVALANCHE_INTENSITY = 0.90              # ABBM intergranular collective avalanche intensity
ALPHA_COUP          = 0.15              # Electromagnetic feedback coupling factor (curl M)
MAG_AXIS            = (1.0, 0.0, 0.0)   # Principal axis for magnetization and measurement (longitudinal X)
RNG_SEED            = 42                # Reproducible seed for the microstructure

# --- 4. HIGH TEMPORAL RESOLUTION (2 FULL CYCLES AT 60 kHz SAMPLING) ---
FREQ        = 50.0                       # 50 Hz (T = 20 ms)
STEPS_CYCLE = 1200                       # 1200 steps per cycle (dt = 16.67 us)
DT          = (1.0 / FREQ) / STEPS_CYCLE
NUM_CYCLES  = 2.0                        # 2 full cycles (2400 steps)
NUM_STEPS   = int(STEPS_CYCLE * NUM_CYCLES)
SAVE_EVERY  = 240


def run_barkhausen_gpu(filename):
    print("=========================================================================")
    print("  STARTING TRANSIENT SOLVER COUPLED WITH 3D BARKHAUSEN (C++/CUDA)")
    print("=========================================================================")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
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
    tag_map = np.zeros(int(np.max(node_tags_np)) + 1, dtype=np.int32)
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
    pids_np     = np.concatenate(pids_list).astype(np.int32)
    num_elements = len(elements_2d)
    num_sample_elems = int(np.sum(pids_np == ID_SAMPLE))

    unique_coil_nodes = np.unique(np.concatenate(coil_node_indices))
    coil_center = np.mean(nodes[unique_coil_nodes], axis=0)

    print(f"  -> Mesh loaded: {num_nodes:,} nodes | {num_elements:,} total tetrahedrons")
    print(f"  -> Elements in Sample (ID={ID_SAMPLE}): {num_sample_elems:,} tetrahedrons "
          f"({num_sample_elems * NUM_DOMAINS:,} micro-domains in GPU)")

    num_dofs = num_nodes * 3
    K_eff = fem_simulator.MatrixCSR(num_dofs, num_dofs)
    M     = fem_simulator.MatrixCSR(num_dofs, num_dofs)
    fem_simulator.assemble_topology_elements(K_eff, elements_np)
    fem_simulator.assemble_topology_elements(M, elements_np)

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

    f0_np = np.zeros(num_dofs, dtype=np.float32)
    fem_simulator.assemble_transient(
        K_eff, M, f0_np, elements_np,
        inv_mu_per_elem, sigma_per_elem,
        Jx_per_elem, Jy_per_elem, Jz_per_elem,
        MU0, float(DT), nodes_x, nodes_y, nodes_z
    )

    fem_simulator.apply_dirichlet_bc(K_eff, nodes_x, nodes_y, nodes_z, tol=1e-3, penalty=1e20)

    print(f"  -> Executing {NUM_STEPS} integration steps on GPU (dt = {DT*1e6:.2f} us)...")
    t0 = time.perf_counter()
    A_snapshots, B_hist, M_hist, dMdt_hist = fem_simulator.solve_transient_barkhausen_gpu(
        K_eff, M, f0_np,
        elements_np, pids_np, ID_SAMPLE,
        SENSOR_BOUNDS,
        nodes_x, nodes_y, nodes_z,
        inv_mu_per_elem, sigma_per_elem, MU0, float(DT),
        NUM_STEPS, float(FREQ), SAVE_EVERY,
        NUM_DOMAINS, float(G_CONST), float(H_PARAM),
        float(MSAT), float(MEAN_HC), float(STD_HC),
        float(H_ANH), float(CORR_LENGTH), float(AVALANCHE_INTENSITY),
        float(ALPHA_COUP), MAG_AXIS, RNG_SEED,
        tol=5e-5, max_iter=600
    )
    elapsed = time.perf_counter() - t0
    print(f"  -> Simulation completed in {elapsed:.2f} seconds!")

    return B_hist, M_hist, dMdt_hist


out_dir = project_root / "python" / "output"
out_dir.mkdir(parents=True, exist_ok=True)
npz_cache = out_dir / "barkhausen_data.npz"

res = run_barkhausen_gpu(mesh_filename)
if res is None:
    exit()

B_hist, M_hist, dMdt_hist = res
np.savez(str(npz_cache), B_hist=B_hist, M_hist=M_hist, dMdt_hist=dMdt_hist, dt=DT, freq=FREQ, j_mag=J_MAGNITUDE)
print(f"  -> Raw data backed up to: {npz_cache}")

if SENSOR_BOUNDS is not None:
    time_axis = np.arange(len(M_hist)) * DT
    I_exc_plot = -J_MAGNITUDE * np.sin(2.0 * np.pi * FREQ * time_axis)

    start_idx = NUM_STEPS - STEPS_CYCLE

    window = 35
    kernel_smooth = np.hanning(window)
    kernel_smooth /= np.sum(kernel_smooth)
    dMdt_smooth_env = np.convolve(dMdt_hist, kernel_smooth, mode='same')
    mbn_bursts = dMdt_hist - dMdt_smooth_env

    fig = plt.figure(figsize=(15.5, 10))

    # 1. Hysteresis Loop B_x vs Excitation in Sensor Zone
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(I_exc_plot[start_idx:] * 1e-6, B_hist[start_idx:], color='darkblue', lw=2.2)
    ax1.axhline(0, color='gray', lw=0.8, ls='--')
    ax1.axvline(0, color='gray', lw=0.8, ls='--')
    ax1.set_title("1. Hysteresis Loop in Sensor Zone ($B_x$ vs $-J_{\\text{coil}}$)")
    ax1.set_xlabel("Equivalent Current Density $-J$ ($\\text{MA/m}^2$)")
    ax1.set_ylabel("Magnetic Field in Sample $B_x$ (T)")
    ax1.grid(True, alpha=0.3)

    # 2. Domain Wall Hysteresis Cycle M_x in Sensor Zone
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(I_exc_plot[start_idx:] * 1e-6, M_hist[start_idx:] * 1e-3, color='darkgreen', lw=2.2)
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    ax2.axvline(0, color='gray', lw=0.8, ls='--')
    ax2.set_title("2. Domain Wall Hysteresis ($M_x$ vs $-J_{\\text{coil}}$)")
    ax2.set_xlabel("Equivalent Current Density $-J$ ($\\text{MA/m}^2$)")
    ax2.set_ylabel("Irreversible Magnetization $M_x$ (kA/m)")
    ax2.grid(True, alpha=0.3)

    # 3. Magnetic Barkhausen Noise Signal with Double Y-Axis
    ax3 = plt.subplot(2, 1, 2)
    t_ms = time_axis[start_idx:] * 1e3

    l1 = ax3.plot(t_ms, dMdt_hist[start_idx:], color='crimson', alpha=0.45, lw=1.2,
                  label="Total Derivative $dM_x/dt$ (Envelope + Avalanches)")
    ax3.set_xlabel("Time $t$ (ms)")
    ax3.set_ylabel("Total Rate $dM_x/dt$ (A/(m$\\cdot$s))", color='crimson')
    ax3.tick_params(axis='y', labelcolor='crimson')
    ax3.grid(True, alpha=0.3)

    ax3_mbn = ax3.twinx()
    l2 = ax3_mbn.plot(t_ms, mbn_bursts[start_idx:], color='navy', lw=0.75, alpha=0.9,
                      label="Barkhausen Noise Pulses ($V_{\\text{MBN}}$ on Surface Sensor)")
    ax3_mbn.set_ylabel("MBN Burst Amplitude (A/(m$\\cdot$s))", color='navy')
    ax3_mbn.tick_params(axis='y', labelcolor='navy')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right")
    ax3.set_title("3. Magnetic Barkhausen Noise Signal on Sample Surface Sensor (ID = 2)")

    plt.tight_layout()
    out_file = out_dir / "barkhausen_sample_results_hd.png"
    plt.savefig(str(out_file), dpi=220)
    print(f"\nHigh resolution plot successfully saved at: {out_file}!")