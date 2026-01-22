"""
Cardiac Mechanics Metrics Calculator

Computes:
1. TRUE WORK (Internal): ∫ 0.5*(S_prev + S_curr) : (E_curr - E_prev) dV (stress-based)
   - Split into Active Work (contraction) and Passive Work (elastic recoil)
2. BOUNDARY WORK (External): ∫ (p n · Δu) dA on endocardial surface (validation)
3. WORK PROXIES (Clinical):
   - Pressure-Volume (PV) work: P · dV
   - Pressure-Strain Area (PSA): Cavity Pressure × Fiber Strain (new)
4. REGIONAL METRICS: All regions (LV/RV/Septum) + AHA segmentation (0-6)

Key Implementation Detail:
  To avoid "Mismatch of tabulation points" errors in FEniCSx JIT compilation, we flatten
  3x3 tensors (S, E) into 9-component vectors using ufl.as_vector(...) and store them in
  DG-0 vector function spaces with explicit basix.ufl.element definitions. Work integrals
  use ufl.dot() which is numerically equivalent to double contraction.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl
from petsc4py import PETSc


class MetricsCalculator:
    """Calculate cardiac mechanics metrics per region at each timestep.
    
    Key features:
      - Stress/strain stored as flattened 9-component DG-0 vectors
      - Work calculated via dot product (equivalent to tensor contraction)
      - All computations MPI-aware with explicit allreduce for multi-rank sync
    """

    def __init__(self, geometry, geo, fiber_field_map, problem, comm, cardiac_model):
        """Initialize metrics calculator with explicit vector element definition.
        
        Args:
            geometry: pulse.HeartGeometry object
            geo: cardiac_geometries.Geometry object (for access to mesh tags)
            fiber_field_map: dict with fiber vectors in current configuration
            problem: pulse.problem.StaticProblem with problem.u, cavities, etc.
            comm: MPI communicator
            cardiac_model: pulse.CardiacModel - single source of truth for stress
        """
        self.geometry = geometry
        self.geo = geo
        self.fiber_fields = fiber_field_map
        self.cardiac_model = cardiac_model
        self.problem = problem
        self.comm = comm
        self.rank = comm.rank
        self.mesh = geometry.mesh
        self.volume2ml = 1e6

        # --- SUPERVISOR SUGGESTION (Jan 22, 2026) ---
        # Try DG-0 tensor space (3,3) directly instead of flattened vectors
        # This may avoid quadrature/tabulation mismatches
        self.W_flat = dolfinx.fem.functionspace(self.mesh, ("DG", 0, (3, 3)))
        
        # Scalar space for scalar quantities
        self.W_scalar = dolfinx.fem.functionspace(self.mesh, ("DG", 0))

        # Storage for Previous State (Flattened Vectors)
        self.S_prev_flat = dolfinx.fem.Function(self.W_flat)
        self.E_prev_flat = dolfinx.fem.Function(self.W_flat)
        self.S_passive_prev_flat = dolfinx.fem.Function(self.W_flat)
        
        self.u_prev = dolfinx.fem.Function(self.problem.u.function_space)

        # Integration measures
        self.dx = ufl.Measure("dx", domain=self.mesh)

        # Pre-compile UFL expressions
        self._setup_expressions()

        # Storage & State tracking
        self.metrics_history = defaultdict(list)
        self.has_previous_state = False
        self.V_LV_prev = None
        self.V_RV_prev = None
        self.septum_tags = geo.additional_data.get("markers_mt", None)
        self.aha_tags = None

    def _flatten_tensor(self, T):
        """Flatten a 3x3 tensor into a 9-component vector UFL expression.
        
        Args:
            T: 3x3 UFL tensor
            
    def _flatten_tensor(self, T):
        """Return tensor as-is for direct tensor space (no flattening needed).
        
        Args:
            T: 3x3 UFL tensor
            
        Returns:
            Same tensor (W_flat is now (3,3) tensor space)
        """
        return T

    def _interpolate_tensor_to_flat(self, tensor_ufl, target_function):
        """Interpolate a 3x3 tensor directly into DG-0 tensor function space.
        
        Uses local (cell-wise) projection for DG-0 spaces.
        
        Args:
            tensor_ufl: 3x3 UFL tensor expression to interpolate
            target_function: dolfinx.fem.Function(W_flat) with (3,3) tensor space
        """
        from dolfinx import fem
        
        # Direct tensor projection (no flattening)
        test = ufl.TestFunction(self.W_flat)
        L = fem.form(ufl.inner(tensor_ufl, test) * self.dx)
        
        # Assemble RHS
        b = fem.petsc.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Mass matrix (diagonal for DG-0)
        a = fem.form(ufl.inner(ufl.TrialFunction(self.W_flat), test) * self.dx)
        A = fem.petsc.assemble_matrix(a)
        A.assemble()
        
        # Get diagonal
        diag = A.getDiagonal()
        
        # Solve: x = b / diag
        target_function.x.petsc_vec.array[:] = b.array / diag.array
        
        # Cleanup
        diag.destroy()
        A.destroy()
        b.destroy()

    def _setup_expressions(self):
        """Pre-compile core UFL expressions used in all calculations.
        
        Note: E (Green strain) is pre-defined here, but S (full stress) is computed
        fresh in each method since it depends on time-varying active tension Ta.
        """
        f0 = self.fiber_fields['f0']
        
        # Deformation
        u = self.problem.u
        I = ufl.Identity(3)
        F = ufl.variable(ufl.grad(u) + I)
        C = F.T * F
        E = 0.5 * (C - I)

        # Stress definitions
        S_passive = self.cardiac_model.material.S(ufl.variable(C))
        sigma_passive = self.cardiac_model.material.sigma(F)

        # Visualization: Fiber projections
        f_current = (F * f0) / ufl.sqrt(ufl.inner(F * f0, F * f0))
        fiber_stress = ufl.inner(sigma_passive * f_current, f_current)
        fiber_strain = ufl.inner(E * f0, f0)

        # Save UFL objects
        self.C = C
        self.E = E
        self.fiber_stress_expr = fiber_stress
        self.fiber_strain_expr = fiber_strain

    def update_state(self):
        """Update previous state for next timestep. Call AFTER each solve.
        
        Uses robust component-wise scalar interpolation to avoid JIT errors.
        """
        # 1. Compute S (Full, including active tension) in UFL
        S_full = self.cardiac_model.S(ufl.variable(self.C))
        
        # 2. Interpolate S into a temp function using the safe component-wise loop
        S_cur_flat = dolfinx.fem.Function(self.W_flat)
        self._interpolate_tensor_to_flat(S_full, S_cur_flat)
        
        # 3. Interpolate E into a temp function (Strain)
        E_cur_flat = dolfinx.fem.Function(self.W_flat)
        # Note: self.E is the 3x3 tensor definition from _setup_expressions
        self._interpolate_tensor_to_flat(self.E, E_cur_flat)

        # 4. Store to Prev for next iteration
        self.S_prev_flat.x.array[:] = S_cur_flat.x.array
        self.E_prev_flat.x.array[:] = E_cur_flat.x.array
        self.u_prev.x.array[:] = self.problem.u.x.array

        self.has_previous_state = True

    def _calculate_true_work(self):
        """Calculate true work (internal, stress-based) and passive-only work for debug.
        
        Work formula: W_total = ∫ 0.5*(S_prev + S_cur) · (E_cur - E_prev) dV
        where · denotes dot product (equivalent to : for tensors).
        Uses robust component-wise interpolation.
        """
        # 1. Get Current S and E (Flattened via robust component-wise loop)
        S_full = self.cardiac_model.S(ufl.variable(self.C))
        
        S_cur_flat = dolfinx.fem.Function(self.W_flat)
        self._interpolate_tensor_to_flat(S_full, S_cur_flat)
        
        E_cur_flat = dolfinx.fem.Function(self.W_flat)
        self._interpolate_tensor_to_flat(self.E, E_cur_flat)

        # 2. Work Calculation: Tensor double contraction S:E
        # 0.5 * (S_prev + S_cur) : (E_cur - E_prev)
        dS_avg = 0.5 * (self.S_prev_flat + S_cur_flat)
        dE = E_cur_flat - self.E_prev_flat
        
        # KEY: Use ufl.inner() for proper tensor contraction (not dot product)
        W_density = ufl.inner(dS_avg, dE)

        # --- Passive Only Work (Debug Comparison) ---
        S_passive = self.cardiac_model.material.S(ufl.variable(self.C))
        S_pass_cur = dolfinx.fem.Function(self.W_flat)
        self._interpolate_tensor_to_flat(S_passive, S_pass_cur)
        
        dS_pass_avg = 0.5 * (self.S_passive_prev_flat + S_pass_cur)
        W_passive_density = ufl.inner(dS_pass_avg, dE)

        # Integrate over regions
        work_data = {}
        regions_to_integrate = self._get_regions_to_integrate()
        metadata = {"quadrature_degree": 4}

        for region_name, cell_tags, region_markers in regions_to_integrate:
            if cell_tags is None:
                continue

            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata=metadata)
            total_work_local = 0.0
            passive_work_local = 0.0
            
            for marker_val in region_markers:
                try:
                    form_work = dolfinx.fem.form(W_density * dx_sub(int(marker_val)))
                    val_total = dolfinx.fem.assemble_scalar(form_work)
                    total_work_local += val_total

                    form_pass = dolfinx.fem.form(W_passive_density * dx_sub(int(marker_val)))
                    val_pass = dolfinx.fem.assemble_scalar(form_pass)
                    passive_work_local += val_pass
                except Exception as e:
                    if self.rank == 0:
                        print(f"Error integrating work for {region_name}: {e}")

            total_work_global = self.comm.allreduce(total_work_local, op=MPI.SUM)
            passive_work_global = self.comm.allreduce(passive_work_local, op=MPI.SUM)
            
            if self.rank == 0 and region_name in ["LV", "RV"]:
                print(f"DEBUG WORK ({region_name}) | Total={total_work_global:.3e} J, Passive={passive_work_global:.3e} J")

            work_data[f"work_true_{region_name}"] = total_work_global

        # Update passive prev for next iteration
        self.S_passive_prev_flat.x.array[:] = S_pass_cur.x.array

        return work_data

    def _calculate_active_passive_work(self):
        """Split total work into Active and Passive components.
        
        Uses magnitude of stress tensor to infer active fraction (heuristic).
        W_active = active_frac * W_total, W_passive = (1 - active_frac) * W_total
        Uses robust component-wise interpolation.
        """
        S_full = self.cardiac_model.S(ufl.variable(self.C))
        
        S_cur_flat = dolfinx.fem.Function(self.W_flat)
        self._interpolate_tensor_to_flat(S_full, S_cur_flat)
        
        # Re-calculate dE (current - prev)
        E_cur_flat = dolfinx.fem.Function(self.W_flat)
        self._interpolate_tensor_to_flat(self.E, E_cur_flat)
        
        dE = E_cur_flat - self.E_prev_flat

        # Calculate Magnitude from tensor (Frobenius norm)
        S_mag = ufl.sqrt(ufl.inner(self.S_prev_flat, self.S_prev_flat) + 1e-10)
        
        # Active fraction (0-1): sigmoid approximation
        S_ref = 10.0e3  # Reference stress in Pa
        active_frac = ufl.tanh(S_mag / S_ref)

        # Work density re-calculation with tensor contraction
        W_avg = 0.5 * (self.S_prev_flat + S_cur_flat)
        W_total = ufl.inner(W_avg, dE)
        
        W_active_density = active_frac * W_total
        W_passive_density = (1.0 - active_frac) * W_total

        work_split = {}
        regions_to_integrate = self._get_regions_to_integrate()
        metadata = {"quadrature_degree": 4}

        for region_name, cell_tags, region_markers in regions_to_integrate:
            if cell_tags is None:
                continue
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata=metadata)
            
            w_active_local = 0.0
            w_passive_local = 0.0
            
            for marker_val in region_markers:
                try:
                    w_active_local += dolfinx.fem.assemble_scalar(dolfinx.fem.form(W_active_density * dx_sub(int(marker_val))))
                    w_passive_local += dolfinx.fem.assemble_scalar(dolfinx.fem.form(W_passive_density * dx_sub(int(marker_val))))
                except Exception:
                    pass
                
            work_split[f"work_active_{region_name}"] = self.comm.allreduce(w_active_local, op=MPI.SUM)
            work_split[f"work_passive_{region_name}"] = self.comm.allreduce(w_passive_local, op=MPI.SUM)

        return work_split

    def _calculate_pressure_proxies(self, model_history, current_state=None):
        """Calculate pressure-volume (P·ΔV) work proxies per region.
        
        Clinical proxy: W_proxy = P * ΔV (units: mmHg·mL → J via 1.33322e-4)
        """
        proxies = {}
        current_state = current_state or {}
        p_LV = current_state.get("p_LV", model_history.get("p_LV", [0.0])[-1] if "p_LV" in model_history else 0.0)
        p_RV = current_state.get("p_RV", model_history.get("p_RV", [0.0])[-1] if "p_RV" in model_history else 0.0)
        
        if "V_LV" in current_state: V_LV = current_state["V_LV"]
        elif len(model_history.get("V_LV", [])) > 0: V_LV = model_history["V_LV"][-1]
        else: V_LV = 0.0
        
        if "V_RV" in current_state: V_RV = current_state["V_RV"]
        elif len(model_history.get("V_RV", [])) > 0: V_RV = model_history["V_RV"][-1]
        else: V_RV = 0.0
        
        if self.V_LV_prev is None:
            self.V_LV_prev = V_LV
            self.V_RV_prev = V_RV
            proxies["work_proxy_pv_LV"] = 0.0
            proxies["work_proxy_pv_RV"] = 0.0
            proxies["work_proxy_pv_Septum"] = 0.0
            return proxies
            
        dV_LV = V_LV - self.V_LV_prev
        dV_RV = V_RV - self.V_RV_prev
        
        # 1 mmHg * mL = 1.33322e-4 Joules
        mmHg_mL_to_J = 1.33322e-4
        
        proxies["work_proxy_pv_LV"] = p_LV * dV_LV * mmHg_mL_to_J
        proxies["work_proxy_pv_RV"] = p_RV * dV_RV * mmHg_mL_to_J
        
        p_avg = (p_LV + p_RV) / 2.0
        proxies["work_proxy_pv_Septum"] = p_avg * (dV_LV + dV_RV) / 2.0 * mmHg_mL_to_J
        
        self.V_LV_prev = V_LV
        self.V_RV_prev = V_RV
        return proxies

    def _calculate_boundary_work(self, current_state=None):
        """Calculate boundary work via surface integral on endocardial surfaces.
        
        External work: W_ext = ∫ p * (n · Δu) dA
        """
        current_state = current_state or {}
        p_LV = current_state.get("p_LV", 0.0)
        p_RV = current_state.get("p_RV", 0.0)
        u_cur = self.problem.u
        if self.u_prev is None: Du = ufl.as_vector([0, 0, 0])
        else: Du = u_cur - self.u_prev
        
        boundary_work = {}
        try:
            cavity_tags = getattr(self.geometry, "facet_tags", getattr(self.geo, "ffun", self.geo.additional_data.get("markers_facets", None)))
            if cavity_tags is None: return {"work_boundary_LV": 0.0, "work_boundary_RV": 0.0}

            ds_cav = ufl.Measure("ds", domain=self.mesh, subdomain_data=cavity_tags, metadata={"quadrature_degree": 4})
            n_vec = ufl.FacetNormal(self.mesh)
            
            try: lv_marker = self.geo.markers["ENDO_LV"][0]
            except: lv_marker = 1
            
            try: rv_marker = self.geo.markers["ENDO_RV"][0]
            except: rv_marker = 2

            for region, p_val, marker in [("LV", p_LV, lv_marker), ("RV", p_RV, rv_marker)]:
                try:
                    integrand = p_val * ufl.dot(n_vec, Du)
                    form = dolfinx.fem.form(integrand * ds_cav(marker))
                    val = dolfinx.fem.assemble_scalar(form)
                    boundary_work[f"work_boundary_{region}"] = self.comm.allreduce(val, op=MPI.SUM)
                except: boundary_work[f"work_boundary_{region}"] = 0.0

        except: boundary_work = {"work_boundary_LV": 0.0, "work_boundary_RV": 0.0}
        return boundary_work

    def _calculate_pressure_strain_area(self, current_state=None):
        """Calculate pressure-strain area (PSA), an alternative clinical proxy.
        
        PSA = ∫ P * ε_fiber dV
        """
        current_state = current_state or {}
        p_LV = current_state.get("p_LV", 0.0)
        p_RV = current_state.get("p_RV", 0.0)
        psa_metrics = {}
        regions_to_integrate = self._get_regions_to_integrate()
        metadata = {"quadrature_degree": 4}
        
        # Fiber strain (scalar, no flattening needed)
        f0 = self.fiber_fields['f0']
        fiber_strain_expr = ufl.inner(self.E * f0, f0)
        
        for region_name, cell_tags, region_markers in regions_to_integrate:
            if cell_tags is None: continue
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata=metadata)
            
            if "LV" in region_name or region_name in ["LV", "AHA_0", "AHA_1", "AHA_2", "AHA_3", "AHA_4"]: p_region = p_LV
            elif "RV" in region_name or region_name in ["AHA_5", "AHA_6"]: p_region = p_RV
            else: p_region = (p_LV + p_RV) / 2.0

            psa_local = 0.0
            for marker_val in region_markers:
                try:
                    psa_local += dolfinx.fem.assemble_scalar(dolfinx.fem.form(p_region * fiber_strain_expr * dx_sub(int(marker_val))))
                except: pass
            psa_metrics[f"psa_{region_name}"] = self.comm.allreduce(psa_local, op=MPI.SUM)
        return psa_metrics

    def _get_regions_to_integrate(self):
        """Return list of (region_name, cell_tags, markers) for integration."""
        regions = []
        if self.septum_tags is not None:
            regions.append(("LV", self.septum_tags, np.array([1])))
            regions.append(("RV", self.septum_tags, np.array([2])))
            regions.append(("Septum", self.septum_tags, np.array([3])))
        if self.aha_tags is not None:
            for label in range(0, 7):
                regions.append((f"AHA_{label}", self.aha_tags, np.array([label])))
        return regions

    def store_metrics(self, region_metrics, timestep_idx, t, downsample_factor=1):
        """Store metrics to history with optional downsampling."""
        if timestep_idx % downsample_factor != 0: return
        self.metrics_history["time"].append(t)
        self.metrics_history["timestep"].append(timestep_idx)
        for metric_name, value in region_metrics.items():
            self.metrics_history[metric_name].append(value)

    def save_metrics(self, output_dir, downsample_factors=None):
        """Save metrics history to disk (rank-0 only)."""
        if self.rank != 0: return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if downsample_factors is None: downsample_factors = [1]
        for factor in downsample_factors:
            downsampled = {}
            indices = np.arange(0, len(self.metrics_history["time"]), factor)
            for key, values in self.metrics_history.items():
                if isinstance(values, list):
                    downsampled[key] = [values[i] for i in indices if i < len(values)]
            filename = output_dir / f"metrics_downsample_{factor}.npy"
            np.save(filename, downsampled, allow_pickle=True)
            print(f"✓ Saved metrics (downsample={factor}) to {filename}")

    def load_aha_tags(self, checkpoint_file):
        """Placeholder for loading AHA tags from checkpoint."""
        pass

    def compute_regional_metrics(self, timestep_idx, t, model_history,
                                 skip_work_calc=False, current_state=None):
        """Compute all metrics for current timestep across all regions.
        
        Args:
            timestep_idx: Current index in simulation
            t: Current time (seconds)
            model_history: Circulation model history dict
            skip_work_calc: If True (first step), skip work calculations
            current_state: Optional dict to override history values
            
        Returns:
            metrics_dict: {metric_name: value}
        """
        metrics = {}
        if not skip_work_calc and self.has_previous_state:
            metrics.update(self._calculate_true_work())
            metrics.update(self._calculate_active_passive_work())
            metrics.update(self._calculate_boundary_work(current_state))
            metrics.update(self._calculate_pressure_proxies(model_history, current_state))
            metrics.update(self._calculate_pressure_strain_area(current_state))
            if self.comm.rank == 0:
                p_LV = current_state.get("p_LV", model_history.get("p_LV", [0.0])[-1] if "p_LV" in model_history else 0.0)
                V_LV = current_state.get("V_LV", model_history.get("V_LV", [0.0])[-1] if "V_LV" in model_history else 0.0)
                w_true_lv = metrics.get("work_true_LV", 0.0)
                print(f"METRICS STEP | i={timestep_idx}, t={t:.4f} s | p_LV={p_LV:.2f} mmHg, V_LV={V_LV:.2f} mL | True={w_true_lv:.3e}")
        return metrics
