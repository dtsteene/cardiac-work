
import dolfinx
import dolfinx.fem
import ufl
import numpy as np
from mpi4py import MPI
from pathlib import Path

def test_metrics_bug():
    comm = MPI.COMM_WORLD
    
    # Create a unit cube mesh (1x1x1 meters)
    mesh = dolfinx.mesh.create_unit_cube(comm, 2, 2, 2)
    
    # 1. Simulate the "True Work" components
    # DG1 Tensor Space
    W_tensor = dolfinx.fem.functionspace(mesh, ("DG", 1, (3, 3)))
    
    # Create a constant Stress field (Identity * 1000 Pa)
    S_cur = dolfinx.fem.Function(W_tensor)
    S_prev = dolfinx.fem.Function(W_tensor)
    
    # Set S_cur to Identity (1 kPa) manually
    # Identity flattened: 1,0,0, 0,1,0, 0,0,1
    # We want this for EVERY dof.
    num_dofs = W_tensor.dofmap.index_map.size_local * W_tensor.dofmap.index_map_bs
    # This might be tricky if block size handles the tensor dim.
    # W_tensor bs is 9?
    
    # Safer way: Use Expression with valid domain
    I_const = dolfinx.fem.Constant(mesh, np.array([[1000,0,0],[0,1000,0],[0,0,1000]], dtype=float))
    S_expr = dolfinx.fem.Expression(I_const, W_tensor.element.interpolation_points)
    S_cur.interpolate(S_expr)
    
    # S_prev is 0
    S_prev.x.array[:] = 0.0
    
    # Create Strain Step (Uniform expansion of 1%)
    E_cur = dolfinx.fem.Function(W_tensor)
    E_prev = dolfinx.fem.Function(W_tensor)
    
    E_const = dolfinx.fem.Constant(mesh, np.array([[0.01,0,0],[0,0.01,0],[0,0,0.01]], dtype=float))
    E_expr = dolfinx.fem.Expression(E_const, W_tensor.element.interpolation_points)
    E_cur.interpolate(E_expr)
    E_prev.x.array[:] = 0.0
    
    # Theoretical Work Calculation:
    # W_density = 0.5 * (S_prev + S_cur) : (E_cur - E_prev)
    # W_density = 0.5 * (1000) : (0.01) * 3 (trace of identity)
    # Actually: sum(S_diff * E_diff)
    # Inner(I, I) = 3
    # W_density = 0.5 * 1000 * 0.01 * 3 = 15 J/m^3.
    # Total Volume = 1 m^3.
    # Total Work = 15 Joules.
    
    expected_density = 15.0
    expected_total_work = 15.0
    
    print(f"Expected Density: {expected_density} Pa")
    print(f"Expected Total Work: {expected_total_work} J")
    
    # === REPRODUCE METRICS_CALCULATOR LOGIC ===
    S_cur_arr = S_cur.x.array.reshape((-1, 3, 3))
    S_prev_arr = S_prev.x.array.reshape((-1, 3, 3))
    E_cur_arr = E_cur.x.array.reshape((-1, 3, 3))
    E_prev_arr = E_prev.x.array.reshape((-1, 3, 3))
    
    dE = E_cur_arr - E_prev_arr
    dS_avg = 0.5 * (S_cur_arr + S_prev_arr)
    
    # Scalar work density at each DG1 point
    work_density = np.einsum('...ij,...ij->...', dS_avg, dE)
    
    print(f"Num Cells: {mesh.topology.index_map(mesh.topology.dim).size_local}")
    print(f"Num DOFs (Work Density array size): {len(work_density)}")
    
    # Bugged 'Integration':
    # "cell_indices" in the original code are gathered from markers
    # Here let's assume one region covers the whole mesh (indices 0 to N_cells-1)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_indices = np.arange(num_cells)
    
    try:
        # The Bug: Accessing DOFs using Cell Indices
        flawed_sum = np.sum(work_density[cell_indices])
        
        # Original code divides by volume to get "True Work" (which implies density)
        vol = 1.0
        flawed_result = flawed_sum / vol
        
        print(f"Flawed Logic Result: {flawed_result}")
        print(f"Ratio (Flawed/Expected): {flawed_result / expected_total_work}")
        
    except IndexError:
        print("ERROR: Index out of bounds (as expected if N_cells < N_dofs, but wait...)")
        # DG1 typically has MORE dofs than cells. So accessing 0..N_cells works, but is wrong.
        
    # === CORRECT LOGIC ===
    # Work = Integral(W_density * dx)
    dx = ufl.Measure("dx", domain=mesh)
    
    # Define form using the Functions directly
    # Note: We must project the numpy calculation back OR use UFL directly
    # Using UFL directly is better
    W_fl = ufl.inner(0.5 * (S_cur + S_prev), E_cur - E_prev)
    work_form = dolfinx.fem.form(W_fl * dx)
    correct_integral = dolfinx.fem.assemble_scalar(work_form)
    correct_integral = comm.allreduce(correct_integral, op=MPI.SUM)
    
    print(f"Correct UFL Integration: {correct_integral}")

if __name__ == "__main__":
    test_metrics_bug()
