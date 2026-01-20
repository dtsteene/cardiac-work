
import dolfinx
import dolfinx.io
from mpi4py import MPI
import numpy as np
import ufl
from pathlib import Path

def test_geometry_volume():
    """
    Test 1.1: Calculate Mesh Volume to determine unit system (m vs mm).
    """
    comm = MPI.COMM_WORLD
    
    # Path to existing geometry (using the one from the actual run)
    geo_path = Path("/home/dtsteene/D1/cardiac-work/results/sims/run_942135/geometry/mesh.xdmf")
    
    if not geo_path.exists():
        print(f"Error: Geometry not found at {geo_path}")
        return

    print(f"Loading geometry from {geo_path}...")
    with dolfinx.io.XDMFFile(comm, geo_path, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Mesh")
        
    # Get coordinates range to check bounding box
    coords = mesh.geometry.x
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    dims = max_coords - min_coords
    
    print("\n=== GEOMETRY BOUNDING BOX ===")
    print(f"Min: {min_coords}")
    print(f"Max: {max_coords}")
    print(f"Dimensions: {dims}")
    
    # Heuristic check for units
    # If dimensions are ~ 0.1, it's likely meters.
    # If dimensions are ~ 100, it's likely mm.
    if np.max(dims) > 10:
        print("-> Dimensions suggest units are MILLIMETERS (mm)")
        unit_scale = "mm"
        to_meters = 1e-3
    else:
        print("-> Dimensions suggest units are METERS (m)")
        unit_scale = "m"
        to_meters = 1.0
        
    # Measure Volume
    dx = ufl.Measure("dx", domain=mesh)
    volume_form = dolfinx.fem.form(dolfinx.fem.Constant(mesh, 1.0) * dx)
    local_vol = dolfinx.fem.assemble_scalar(volume_form)
    total_vol = comm.allreduce(local_vol, op=MPI.SUM)
    
    print("\n=== VOLUME ANALYSIS ===")
    print(f"Raw Integrated Volume: {total_vol:.4f} (mesh units cubed)")
    
    if unit_scale == "mm":
        vol_liters = total_vol * (1e-3)**3 * 1000 # mm^3 -> m^3 -> L
        vol_ml = total_vol / 1000.0 # mm^3 = microL. Wait. 1 cm^3 = 1000 mm^3 = 1 mL.
        print(f"Volume in mL (assuming mm): {vol_ml:.4f} mL")
        print(f"Volume in m^3 (assuming mm): {total_vol * 1e-9:.4e} m^3")
    else:
        vol_ml = total_vol * 1e6 # m^3 -> mL
        print(f"Volume in mL (assuming m): {vol_ml:.4f} mL")
    
    print("\n=== EXPECTATION ===")
    print("Normal adult BiV myocardium volume ~ 200-300 mL")
    
    if 100 < vol_ml < 500:
        print(f"-> SUCCESS: Volume {vol_ml:.1f} mL is physiologically reasonable for {unit_scale}.")
    else:
        print(f"-> WARNING: Volume {vol_ml:.1f} mL seems unreasonable.")

if __name__ == "__main__":
    test_geometry_volume()
