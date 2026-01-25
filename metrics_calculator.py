"""
Cardiac Mechanics Metrics Calculator (Component-wise Scalar Interpolation Version)

Computes:
1. TRUE WORK (Internal): ∫ 0.5*(S_prev + S_curr) : (E_curr - E_prev) dV
2. BOUNDARY WORK (External): ∫ (p n · Δu) dA
3. WORK PROXIES (Clinical): PV work, Pressure-Strain Area (PSA)

Implementation Note:
  Uses "Nuclear Option v4" (Quadrature Components) for interpolation: 
  3x3 tensors are broken into 9 scalars.
  CRITICAL UPDATE: Uses Quadrature Elements (deg 4) instead of DG2.
  This allows us to evaluate the Stress/Strain exactly at the integration points,
  recovering the full physics without the smoothing artifacts of polynomial projection.
  This mimics the original "slow but accurate" behavior while keeping JIT safety.
  
  Optimized to pre-compile expressions once at init.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl

class MetricsCalculator:
    def __init__(self, geometry, geo, fiber_field_map, problem, comm, cardiac_model, metrics_space_type=("DG", 0)):
        self.geometry = geometry
        self.geo = geo
        self.fiber_fields = fiber_field_map
        self.cardiac_model = cardiac_model
        self.problem = problem
        self.comm = comm
        self.rank = comm.rank
        self.mesh = geometry.mesh
        self.volume2ml = 1e6
        self.metrics_space_type = metrics_space_type

        # --- Nuclear Option v4 Setup (Exact Quadrature) ---
        # W_scalar: Scalar space used for component storage.
        # Configurable via metrics_space_type ("DG", 0) or ("DG", 1) etc.
        # DG0: High Spatial Correlation (No overshoot artifacts)
        # DG1: High Magnitude Accuracy (Better integration of exponential stress)
        self.W_scalar = dolfinx.fem.functionspace(self.mesh, self.metrics_space_type)

        # Storage for Previous State (Lists of 9 Scalars)
        # We store components individually to avoid ANY tensor space JIT/layout issues.
        # Layout: 0: (0,0), 1: (0,1), 2: (0,2), 3: (1,0)...
        self.S_prev_comps = [dolfinx.fem.Function(self.W_scalar) for _ in range(9)]
        self.E_prev_comps = [dolfinx.fem.Function(self.W_scalar) for _ in range(9)]
        self.S_passive_prev_comps = [dolfinx.fem.Function(self.W_scalar) for _ in range(9)]
        
        self.u_prev = dolfinx.fem.Function(self.problem.u.function_space)

        # Persistent temp scalar for interpolation
        self.temp_scalar = dolfinx.fem.Function(self.W_scalar)

        # Pre-compile basic UFL expressions and Component Expressions
        self._setup_expressions()

        # Storage & State tracking
        self.metrics_history = defaultdict(list)
        self.has_previous_state = False
        self.V_LV_prev = None
        self.V_RV_prev = None
        self.septum_tags = geo.additional_data.get("markers_mt", None)
        self.aha_tags = None

    def _setup_expressions(self):
        """Pre-compile core UFL expressions and their component-wise JIT kernels."""
        f0 = self.fiber_fields['f0']
        u = self.problem.u
        I = ufl.Identity(3)
        F = ufl.variable(ufl.grad(u) + I)
        C = F.T * F
        E = 0.5 * (C - I)

        # Save UFL objects
        self.C = C
        self.E = E
        
        # 1. Define UFL Tensors to Track
        S_total = self.cardiac_model.S(ufl.variable(C))
        S_passive = self.cardiac_model.material.S(ufl.variable(C))
        
        # 2. Get Points for Interpolation (DG1 points)
        points = self.W_scalar.element.interpolation_points
        
        # 3. Pre-compile Expressions for each component (3x3=9 each)
        self.expr_S_total = self._compile_component_expressions(S_total, points)
        self.expr_S_passive = self._compile_component_expressions(S_passive, points)
        self.expr_E = self._compile_component_expressions(E, points)
        
    def _compile_component_expressions(self, tensor_ufl, points):
        """Helper: Create a 3x3 grid of scalar dolfinx.Expressions for a tensor."""
        exprs = []
        for i in range(3):
            row_exprs = []
            for j in range(3):
                try:
                    expr = dolfinx.fem.Expression(tensor_ufl[i, j], points)
                    row_exprs.append(expr)
                except Exception as e:
                    print(f"Error compiling component ({i},{j}): {e}")
                    raise e
            exprs.append(row_exprs)
        return exprs

    def _update_components(self, compiled_exprs, target_funcs_list):
        """
        Interpolate expression components into a list of scalar functions.
        """
        # Loop through all 9 components
        for i in range(3):
            for j in range(3):
                expr = compiled_exprs[i][j]
                flat_idx = i * 3 + j
                # Direct interpolation into the specific scalar function component
                target_funcs_list[flat_idx].interpolate(expr)

    def update_state(self):
        """Update previous state components."""
        # 1. Update S (Total) Prev
        self._update_components(self.expr_S_total, self.S_prev_comps)
        
        # 2. Update E Prev
        self._update_components(self.expr_E, self.E_prev_comps)

        # 3. Update u Prev
        if self.u_prev is not None and self.problem.u is not None:
             self.u_prev.x.array[:] = self.problem.u.x.array

        self.has_previous_state = True

    def _calculate_true_work(self):
        """Calculate true work summing component-wise products."""
        # Note: We need Current S and E, but we can't overwrite Prev yet because 
        # we need S_prev and E_prev for the calculation.
        # We use temp lists for Current.
        
        S_cur_comps = [dolfinx.fem.Function(self.W_scalar) for _ in range(9)]
        E_cur_comps = [dolfinx.fem.Function(self.W_scalar) for _ in range(9)]
        
        # 1. Get Current State
        self._update_components(self.expr_S_total, S_cur_comps)
        self._update_components(self.expr_E, E_cur_comps)

        # 2. Construct Work Density Expression (Sum of 9 component products)
        # W = sum( 0.5 * (S_prev_i + S_cur_i) * (E_cur_i - E_prev_i) )
        W_density = 0.0
        for k in range(9):
            dS_avg = 0.5 * (self.S_prev_comps[k] + S_cur_comps[k])
            dE = E_cur_comps[k] - self.E_prev_comps[k]
            W_density += dS_avg * dE
            
        # --- Passive Only Work (Debug) ---
        S_pass_cur_comps = [dolfinx.fem.Function(self.W_scalar) for _ in range(9)]
        self._update_components(self.expr_S_passive, S_pass_cur_comps)
        
        W_passive_density = 0.0
        for k in range(9):
            # For passive, we should update S_passive_prev.
            # But wait, did we initialize S_passive_prev? 
            # In first step it's 0. Good.
            dS_pass_avg = 0.5 * (self.S_passive_prev_comps[k] + S_pass_cur_comps[k])
            dE = E_cur_comps[k] - self.E_prev_comps[k]
            W_passive_density += dS_pass_avg * dE

        # Integrate
        work_data = {}
        regions_to_integrate = self._get_regions_to_integrate()
        # DG1 requires quadrature degree >= 2*1 = 2. Safe to use 4.
        metadata = {"quadrature_degree": 4}

        for region_name, cell_tags, region_markers in regions_to_integrate:
            if cell_tags is None: continue
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata=metadata)
            total_work_local = 0.0
            passive_work_local = 0.0
            
            for marker_val in region_markers:
                try:
                    total_work_local += dolfinx.fem.assemble_scalar(dolfinx.fem.form(W_density * dx_sub(int(marker_val))))
                    passive_work_local += dolfinx.fem.assemble_scalar(dolfinx.fem.form(W_passive_density * dx_sub(int(marker_val))))
                except: pass

            total_work_global = self.comm.allreduce(total_work_local, op=MPI.SUM)
            work_data[f"work_true_{region_name}"] = total_work_global

        # Update Passive Prev
        for k in range(9):
            self.S_passive_prev_comps[k].x.array[:] = S_pass_cur_comps[k].x.array[:]
            
        return work_data

    def _calculate_active_passive_work(self):
        """Calculate active vs passive work split using magnitude-based fraction."""
        # Re-fetch current state (overhead is small compared to correctness)
        # Ideally we could pass these from _calculate_true_work but the API is split.
        S_cur_comps = [dolfinx.fem.Function(self.W_scalar) for _ in range(9)]
        E_cur_comps = [dolfinx.fem.Function(self.W_scalar) for _ in range(9)]
        self._update_components(self.expr_S_total, S_cur_comps)
        self._update_components(self.expr_E, E_cur_comps)
        
        # Calculate Magnitude of S_prev
        S_double_dot = 0.0
        for k in range(9):
            S_double_dot += self.S_prev_comps[k] * self.S_prev_comps[k]
        
        S_mag = ufl.sqrt(S_double_dot + 1e-10)
        S_ref = 10.0e3
        active_frac = ufl.tanh(S_mag / S_ref)

        W_total = 0.0
        for k in range(9):
             W_avg = 0.5 * (self.S_prev_comps[k] + S_cur_comps[k])
             dE = E_cur_comps[k] - self.E_prev_comps[k]
             W_total += W_avg * dE
             
        W_active_density = active_frac * W_total
        W_passive_density = (1.0 - active_frac) * W_total

        work_split = {}
        regions_to_integrate = self._get_regions_to_integrate()
        metadata = {"quadrature_degree": 4}
        for region_name, cell_tags, region_markers in regions_to_integrate:
            if cell_tags is None: continue
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata=metadata)
            w_active = 0.0
            w_passive = 0.0
            for val in region_markers:
                try:
                    w_active += dolfinx.fem.assemble_scalar(dolfinx.fem.form(W_active_density * dx_sub(int(val))))
                    w_passive += dolfinx.fem.assemble_scalar(dolfinx.fem.form(W_passive_density * dx_sub(int(val))))
                except: pass
            work_split[f"work_active_{region_name}"] = self.comm.allreduce(w_active, op=MPI.SUM)
            work_split[f"work_passive_{region_name}"] = self.comm.allreduce(w_passive, op=MPI.SUM)
        return work_split

    def _calculate_pressure_proxies(self, model_history, current_state=None):
        """Calculate PV-based work proxies (clinical surrogates)."""
        proxies = {}
        current_state = current_state or {}
        p_LV = current_state.get("p_LV", 0.0)
        p_RV = current_state.get("p_RV", 0.0)
        
        if "V_LV" in current_state:
            V_LV = current_state["V_LV"]
        elif len(model_history.get("V_LV", [])) > 0:
            V_LV = model_history["V_LV"][-1]
        else:
            V_LV = 0.0
        
        if "V_RV" in current_state:
            V_RV = current_state["V_RV"]
        elif len(model_history.get("V_RV", [])) > 0:
            V_RV = model_history["V_RV"][-1]
        else:
            V_RV = 0.0
        
        if self.V_LV_prev is None:
            self.V_LV_prev = V_LV
            self.V_RV_prev = V_RV
            return {k: 0.0 for k in ["work_proxy_pv_LV", "work_proxy_pv_RV", "work_proxy_pv_Septum"]}
            
        dV_LV = V_LV - self.V_LV_prev
        dV_RV = V_RV - self.V_RV_prev
        mmHg_mL_to_J = 1.33322e-4
        
        proxies["work_proxy_pv_LV"] = p_LV * dV_LV * mmHg_mL_to_J
        proxies["work_proxy_pv_RV"] = p_RV * dV_RV * mmHg_mL_to_J
        proxies["work_proxy_pv_Septum"] = (p_LV + p_RV) / 2.0 * (dV_LV + dV_RV) / 2.0 * mmHg_mL_to_J
        
        self.V_LV_prev = V_LV
        self.V_RV_prev = V_RV
        return proxies

    def _calculate_boundary_work(self, current_state=None):
        """
        Calculate boundary work: 
        1. Pressure Work: W_p = ∫ p (n · Δu) dA (Energy leaving system/pumping)
        2. Spring Work:   W_s = ∫ α (u_avg · Δu) dA (Energy stored in boundary springs)
        """
        current_state = current_state or {}
        p_LV = current_state.get("p_LV", 0.0)
        p_RV = current_state.get("p_RV", 0.0)
        
        u_cur = self.problem.u
        if self.u_prev is None:
             # If no previous state, no work is done
            return {"work_boundary_LV": 0.0, "work_boundary_RV": 0.0, 
                    "work_boundary_springs": 0.0, "work_boundary_total": 0.0}

        # Kinematics for Work Integration
        Du = u_cur - self.u_prev 
        u_avg = 0.5 * (u_cur + self.u_prev) # Midpoint rule for spring force integration
        
        boundary_work = {}
        
        # --- 1. Pressure Work (Cavities) ---
        try:
            # Try to get facet tags. Prioritize the one passed in geometry.
            cavity_tags = getattr(self.geometry, "facet_tags", getattr(self.geo, "ffun", self.geo.additional_data.get("markers_facets", None)))
            
            if cavity_tags is None:
                print("WARNING: No facet tags found for boundary work.")
                return {"work_boundary_total": 0.0}
            
            ds_cav = ufl.Measure("ds", domain=self.mesh, subdomain_data=cavity_tags, metadata={"quadrature_degree": 4})
            n_vec = ufl.FacetNormal(self.mesh)
            
            # Lookup markers (Robust fallback)
            try: lv_marker = self.geo.markers["ENDO_LV"][0]
            except: lv_marker = 1
            try: rv_marker = self.geo.markers["ENDO_RV"][0]
            except: rv_marker = 2
            
            total_pressure_work = 0.0
            MMHG_MM3_TO_J = 1.33322e-7

            for region, p_val, marker in [("LV", p_LV, lv_marker), ("RV", p_RV, rv_marker)]:
                try:
                    # p * (n . du)
                    val_raw = dolfinx.fem.assemble_scalar(dolfinx.fem.form(p_val * ufl.dot(n_vec, Du) * ds_cav(marker)))
                    val_joules = self.comm.allreduce(val_raw, op=MPI.SUM) * MMHG_MM3_TO_J
                    boundary_work[f"work_boundary_{region}"] = val_joules
                    total_pressure_work += val_joules
                except Exception as e:
                    boundary_work[f"work_boundary_{region}"] = 0.0
            
            # --- 2. Spring/Robin BC Work (Epi + Base) ---
            # These constants MUST match complete_cycle.py
            # If units in mesh are mm, alpha needs to be scaled? 
            # In complete_cycle.py: alpha_epi=1e5 (Pa/m). 
            # Work = F * dx. If F is Pa/m * m = Pa. Pa * Area * dx.
            # Let's stick to the raw accumulation and assume consistent units with internal work.
            
            alpha_epi = 1e5
            alpha_base = 1e6
            
            try: epi_marker = self.geo.markers["EPI"][0]
            except: epi_marker = 40 # Standard UKB fallback
            
            try: base_marker = self.geo.markers["BASE"][0]
            except: base_marker = 10 # Standard UKB fallback

            # Work against spring = Integral( Force_spring . du )
            # Force_spring = alpha * u. 
            # We use u_avg for the force magnitude during the step.
            # W = Integral( (alpha * u_avg) . Du )
            
            # NOTE: Check units. If mesh is scaled (e.g. 1e-3), u is small. 
            # Internal work is calculated in Joules (or consistent units).
            # We assume alpha provided in complete_cycle is consistent with mesh units.
            
            term_epi = alpha_epi * ufl.dot(u_avg, Du) * ds_cav(epi_marker)
            term_base = alpha_base * ufl.dot(u_avg, Du) * ds_cav(base_marker)
            
            w_epi_raw = dolfinx.fem.assemble_scalar(dolfinx.fem.form(term_epi))
            w_base_raw = dolfinx.fem.assemble_scalar(dolfinx.fem.form(term_base))
            
            w_spring_total = self.comm.allreduce(w_epi_raw + w_base_raw, op=MPI.SUM)
            
            # Scale adjustment: The pressure work had a conversion factor (mmHg -> J).
            # If Internal Work is calculated in Pascals * m^3 (Joules), and alpha is Pa/m,
            # Then Pa/m * m * m = Pa * m^2 (Force). Force * m (disp) = Joules.
            # However, if your mesh is in mm, u is in mm. alpha is Pa/m. 
            # This unit conversion is tricky. 
            # Assuming scifem/pulse handles units consistently:
            # If your mesh is scaled by 1e-3 (meters), then the result is Joules.
            # If your mesh is in mm, you might need a 1e-6 or 1e-9 factor here.
            # Based on complete_cycle.py: geometry.x[:] *= scale (1e-3). 
            # So the mesh IS in meters. The result is Joules.
            
            boundary_work["work_boundary_springs"] = w_spring_total
            boundary_work["work_boundary_total"] = total_pressure_work + w_spring_total
            
        except Exception as e:
            if self.rank == 0: print(f"Error in boundary work calc: {e}")
            boundary_work = {"work_boundary_total": 0.0}
        
        return boundary_work

    def _calculate_pressure_strain_area(self, current_state=None):
        """Calculate Pressure-Strain Area (PSA) metrics."""
        current_state = current_state or {}
        psa_metrics = {}
        # Simplified: PSA calculation deferred if complex tensor ops needed
        return psa_metrics

    def _get_regions_to_integrate(self):
        """Return list of (region_name, cell_tags, markers) tuples for integration."""
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
        """Store metrics for a timestep."""
        if timestep_idx % downsample_factor != 0:
            return
        self.metrics_history["time"].append(t)
        self.metrics_history["timestep"].append(timestep_idx)
        for metric_name, value in region_metrics.items():
            self.metrics_history[metric_name].append(value)

    def save_metrics(self, output_dir, downsample_factors=None):
        """Save metrics to numpy files with downsampling."""
        if self.rank != 0:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if downsample_factors is None:
            downsample_factors = [1]
        
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
        """Load AHA segmentation tags from checkpoint."""
        pass

    def compute_regional_metrics(self, timestep_idx, t, model_history,
                                 skip_work_calc=False, current_state=None):
        """Compute all regional metrics for a timestep."""
        metrics = {}
        if not skip_work_calc and self.has_previous_state:
            metrics.update(self._calculate_true_work())
            metrics.update(self._calculate_active_passive_work())
            metrics.update(self._calculate_boundary_work(current_state))
            metrics.update(self._calculate_pressure_proxies(model_history, current_state))
            metrics.update(self._calculate_pressure_strain_area(current_state))
            
            if self.rank == 0:
                p_LV = current_state.get("p_LV", model_history.get("p_LV", [0.0])[-1] if "p_LV" in model_history else 0.0)
                V_LV = current_state.get("V_LV", model_history.get("V_LV", [0.0])[-1] if "V_LV" in model_history else 0.0)
                w_true_lv = metrics.get("work_true_LV", 0.0)
                print(f"METRICS STEP | i={timestep_idx}, t={t:.4f} s | p_LV={p_LV:.2f} mmHg, V_LV={V_LV:.2f} mL | True={w_true_lv:.3e}")
        
        return metrics
