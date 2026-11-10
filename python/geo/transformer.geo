// ====================================================================
// Single-Phase Shell-Type Transformer: High Fidelity for 3D FEM
// Includes: Insulating bobbin, separated windings, and fluid ducts
// ====================================================================
SetFactory("OpenCASCADE");

// 1. CORE PARAMETERS (Units in meters)
// --------------------------------------------------------------------
wc = 0.10; // Central leg width
wo = 0.05; // Outer legs width
hy = 0.05; // Yokes height
ww = 0.08; // Windows width (Total available space = 80mm)
hw = 0.20; // Windows height
d  = 0.10; // Core depth

// Total core dimensions
W = 2*wo + 2*ww + wc;
H = 2*hy + hw;
D = d;

// 2. WINDINGS AND INSULATION PARAMETERS
// --------------------------------------------------------------------
// Building outward from the core in the 80mm window:

g_c  = 0.002; // Mechanical gap Core-Bobbin (2mm per side for coolant flow)
t_b  = 0.003; // Insulating bobbin thickness (fiberglass/pressboard)
t_w1 = 0.020; // Radial thickness of Primary winding (Low Voltage)
g_d  = 0.010; // Main cooling duct between Primary and Secondary (10mm)
t_w2 = 0.020; // Radial thickness of Secondary winding (High Voltage)

// Occupied space is: 2 + 3 + 20 + 10 + 20 = 55mm. 
// 25mm remain free for "fluid" until touching the outer leg.

// 3. VOLUME CONSTRUCTION
// --------------------------------------------------------------------
// --- A. CORE ---
Box(1) = {-W/2, -H/2, -D/2, W, H, D};
Box(2) = {-wc/2 - ww, -hw/2, -D/2 - 0.01, ww, hw, D + 0.02}; 
Box(3) = { wc/2,      -hw/2, -D/2 - 0.01, ww, hw, D + 0.02}; 
BooleanDifference(4) = {Volume{1}; Delete;}{Volume{2,3}; Delete;}; // Vol 4 = Core

// --- B. INSULATING BOBBIN ---
// Hollow tube surrounding the central leg
W_in_b = wc + 2*g_c;    D_in_b = d + 2*g_c;     H_b = hw - 0.004; // 2mm free top/bottom
W_out_b = W_in_b + 2*t_b; D_out_b = D_in_b + 2*t_b;
Box(5) = {-W_out_b/2, -H_b/2, -D_out_b/2, W_out_b, H_b, D_out_b}; // Exterior
Box(6) = {-W_in_b/2,  -H_b/2, -D_in_b/2,  W_in_b,  H_b, D_in_b};  // Interior
BooleanDifference(7) = {Volume{5}; Delete;}{Volume{6}; Delete;};  // Vol 7 = Bobbin

// --- C. PRIMARY WINDING (Low Voltage) ---
// Wound directly over the insulating bobbin
W_in_w1 = W_out_b;      D_in_w1 = D_out_b;      H_w = hw - 0.020; // 10mm free top/bottom
W_out_w1 = W_in_w1 + 2*t_w1; D_out_w1 = D_in_w1 + 2*t_w1;
Box(8) = {-W_out_w1/2, -H_w/2, -D_out_w1/2, W_out_w1, H_w, D_out_w1};
Box(9) = {-W_in_w1/2,  -H_w/2, -D_in_w1/2,  W_in_w1,  H_w, D_in_w1};
BooleanDifference(10) = {Volume{8}; Delete;}{Volume{9}; Delete;}; // Vol 10 = Primary

// --- D. SECONDARY WINDING (High Voltage) ---
// Leave space for the cooling duct (g_d)
W_in_w2 = W_out_w1 + 2*g_d; D_in_w2 = D_out_w1 + 2*g_d;
W_out_w2 = W_in_w2 + 2*t_w2; D_out_w2 = D_in_w2 + 2*t_w2;
Box(11) = {-W_out_w2/2, -H_w/2, -D_out_w2/2, W_out_w2, H_w, D_out_w2};
Box(12) = {-W_in_w2/2,  -H_w/2, -D_in_w2/2,  W_in_w2,  H_w, D_in_w2};
BooleanDifference(13) = {Volume{11}; Delete;}{Volume{12}; Delete;}; // Vol 13 = Secondary

// --- E. FLUID DOMAIN (Enclosure and ducts) ---
W_air = W * 3; H_air = H * 3; D_air = D * 3;
Box(14) = {-W_air/2, -H_air/2, -D_air/2, W_air, H_air, D_air};

// By subtracting ALL solids, gaps and spaces become the fluid
BooleanDifference(15) = {Volume{14}; Delete;}{Volume{4, 7, 10, 13};}; // Vol 15 = Fluid

// Topological Synchronization (Critical!)
BooleanFragments{ Volume{4, 7, 10, 13, 15}; Delete; }{}

// 4. PHYSICAL GROUPS
// --------------------------------------------------------------------
Physical Volume("Iron_Core", 100) = {4};
Physical Volume("Solid_Insulator", 200) = {7};
Physical Volume("Primary_Coil", 300) = {10};
Physical Volume("Secondary_Coil", 400) = {13};
Physical Volume("Coolant_Fluid", 500) = {15}; // Includes exterior and internal ducts

eps = 1e-3;
Physical Surface("Dirichlet_Boundary", 600) = { 
    Surface In BoundingBox{-W_air/2-eps, -H_air/2-eps, -D_air/2-eps, 
                            W_air/2+eps,  H_air/2+eps,  D_air/2+eps} 
};

// 5. MESH CONTROL (Mesh Fields)
// --------------------------------------------------------------------
lc_core = 0.012; // Fine mesh for iron and windings
lc_air  = 0.15;  // Coarse exterior mesh to avoid saturating CSR matrix

Field[1] = Box;
Field[1].VIn = lc_core;
Field[1].VOut = lc_air;
// Refinement over the entire physical volume of the transformer
Field[1].XMin = -W/2 - 0.02;     Field[1].XMax = W/2 + 0.02;
Field[1].YMin = -H/2 - 0.02;     Field[1].YMax = H/2 + 0.02;
Field[1].ZMin = -D_out_w2/2 - 0.02; Field[1].ZMax = D_out_w2/2 + 0.02;
Field[1].Thickness = 0.10; 

Background Field = 1;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;