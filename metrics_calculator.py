import numpy as np
import csv
from pathlib import Path
from collections import defaultdict
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl

class MetricsCalculator:
    def __init__(self, geometry, geo, fiber_field_map, problem, comm, cardiac_model, metrics_space_type=("DG", 0), alpha_epi=1e5, alpha_base=1e6):
        self.geometry = geometry
        self.geo = geo
        self.fiber_fields = fiber_field_map
        self.cardiac_model = cardiac_model
        self.problem = problem
        self.comm = comm
        self.rank = comm.rank
        self.alpha_epi = alpha_epi
        self.alpha_base = alpha_base
        self.mesh = geometry.mesh
        
        # --- 1. Define Function Spaces ---
        element_type = metrics_space_type[0]
        degree = metrics_space_type[1]
        
        # Scalar Space (for Work Density, Strain Energy)
        self.W_scalar = dolfinx.fem.functionspace(self.mesh, (element_type, degree))
        
        # Tensor Space (for Stress S and Strain E)
        self.W_tensor = dolfinx.fem.functionspace(self.mesh, (element_type, degree, (3, 3)))

        # --- 2. Define Functions for State Tracking ---
        self.S_total = dolfinx.fem.Function(self.W_tensor, name="S_total")
        self.S_active = dolfinx.fem.Function(self.W_tensor, name="S_active")
        self.S_passive = dolfinx.fem.Function(self.W_tensor, name="S_passive")
        self.S_comp = dolfinx.fem.Function(self.W_tensor, name="S_compressible")
        
        self.E_cur = dolfinx.fem.Function(self.W_tensor, name="E_cur")
        self.E_prev = dolfinx.fem.Function(self.W_tensor, name="E_prev")
        
        # Initialize Previous State to Zero
        self.E_prev.x.array[:] = 0.0

        # --- 3. Setup UFL Expressions ---
        self._setup_expressions()

        # History and State flags
        self.metrics_history = defaultdict(list)
        self.has_previous_state = False
        self.V_LV_prev = None
        self.V_RV_prev = None
        
        # Strain History for PS Loops
        self.eps_LV_prev = 0.0
        self.eps_RV_prev = 0.0
        self.eps_Septum_prev = 0.0
        
        # Region Tags
        self.region_tags = geo.additional_data.get("markers_mt", None)
        
        # --- 4. Calculate Regional Wall Volumes for Unit Scaling ---
        # We need specific volumes for LV, RV, and Septum to scale the PS Indices correctly.
        self.region_volumes = {}
        try:
            # Helper to integrate volume for a specific tag
            def get_vol(tags):
                dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.region_tags, metadata={"quadrature_degree": 4})
                val = 0.0
                for t in tags:
                     val += dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(self.mesh, 1.0) * dx_sub(int(t))))
                return self.comm.allreduce(val, op=MPI.SUM)

            if self.region_tags:
                self.region_volumes["LV"] = get_vol([1])      # LV Free Wall
                self.region_volumes["RV"] = get_vol([2])      # RV Free Wall
                self.region_volumes["Septum"] = get_vol([3])  # Septum
                self.region_volumes["Whole"] = get_vol([1, 2, 3, 4]) # Whole Mesh
            else:
                 # Fallback if no tags
                 self.region_volumes["LV"] = 1.0
                 self.region_volumes["RV"] = 1.0
                 self.region_volumes["Septum"] = 1.0
                 self.region_volumes["Whole"] = 1.0

            if self.rank == 0:
                 print(f"MetricsCalculator: Volumes calculated.")
                 print(f"  LV Free: {self.region_volumes['LV']:.2e} m3")
                 print(f"  RV Free: {self.region_volumes['RV']:.2e} m3")
                 print(f"  Septum:  {self.region_volumes['Septum']:.2e} m3")

        except Exception as e:
            if self.rank == 0: print(f"MetricsCalculator Warning: Could not calc regional volumes ({e}). Using defaults.")
            self.region_volumes = defaultdict(lambda: 1.0)

    def _setup_expressions(self):
        u = self.problem.u
        I = ufl.Identity(3)
        F = ufl.variable(ufl.grad(u) + I)
        C = ufl.variable(F.T * F) 
        E = 0.5 * (C - I)
        
        S_tot_ufl = self.cardiac_model.S(C)
        S_act_ufl = self.cardiac_model.active.S(C)
        S_pas_ufl = self.cardiac_model.material.S(C)
        S_cmp_ufl = self.cardiac_model.compressibility.S(C)

        points = self.W_tensor.element.interpolation_points
        self.expr_E = dolfinx.fem.Expression(E, points)
        self.expr_S_total = dolfinx.fem.Expression(S_tot_ufl, points)
        self.expr_S_active = dolfinx.fem.Expression(S_act_ufl, points)
        self.expr_S_passive = dolfinx.fem.Expression(S_pas_ufl, points)
        self.expr_S_comp = dolfinx.fem.Expression(S_cmp_ufl, points)

    def update_state(self):
        """Called at the END of a timestep to shift Current -> Prev."""
        self.E_prev.x.array[:] = self.E_cur.x.array[:]
        
        # Track displacement for boundary work
        if not hasattr(self, '_u_prev'):
            self._u_prev = dolfinx.fem.Function(self.problem.u.function_space)
        self._u_prev.x.array[:] = self.problem.u.x.array[:]
        
        self.has_previous_state = True
        
    def _calculate_state_variables(self):
        """
        Calculates absolute state variables (Stress, Strain) at the current time.
        Run this EVERY step (including t=0).
        """
        # --- A. Interpolate Physics ---
        self.E_cur.interpolate(self.expr_E)
        self.S_total.interpolate(self.expr_S_total)
        self.S_active.interpolate(self.expr_S_active)

        # --- B. Integration ---
        data = {}

        # Probe internal active tension
        try:
            data["debug_Ta_internal_max"] = np.max(self.cardiac_model.active.activation.value.x.array)
        except Exception:
            data["debug_Ta_internal_max"] = 0.0
            
        data["debug_S_active_max"] = np.max(self.S_active.x.array)

        regions = self._get_regions_to_integrate()
        f0 = self.fiber_fields['f0']

        # Helper: Project Tensor T onto direction v
        def proj(T, v): return ufl.inner(ufl.dot(T, v), v)

        for region_name, cell_tags, region_markers in regions:
            # Setup Measure
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata={"quadrature_degree": 4})
            
            def assemble_region(expr):
                val = 0.0
                for m in region_markers:
                    form = dolfinx.fem.form(expr * dx_sub(int(m)))
                    val += dolfinx.fem.assemble_scalar(form)
                return self.comm.allreduce(val, op=MPI.SUM)

            # Volume and Averages
            vol = assemble_region(dolfinx.fem.Constant(self.mesh, 1.0))
            
            if vol > 1e-12:
                S_active_mag = ufl.sqrt(ufl.inner(self.S_active, self.S_active))
                data[f"mean_S_ff_{region_name}"] = assemble_region(proj(self.S_total, f0)) / vol
                data[f"mean_E_ff_{region_name}"] = assemble_region(proj(self.E_cur, f0)) / vol
                data[f"mean_S_active_{region_name}"] = assemble_region(S_active_mag) / vol
            else:
                data[f"mean_S_ff_{region_name}"] = 0.0
                data[f"mean_E_ff_{region_name}"] = 0.0
                data[f"mean_S_active_{region_name}"] = 0.0
                
        return data

    def _calculate_incremental_work(self):
        """
        Calculates Work Densities (S : dE).
        Run this only when previous state exists.
        """
        # 1. Update interpolations needed for work (Passive/Comp)
        self.S_passive.interpolate(self.expr_S_passive)
        self.S_comp.interpolate(self.expr_S_comp)
        
        dE = self.E_cur - self.E_prev
        
        # 2. Define Work Densities (UFL)
        wd_total = ufl.inner(self.S_total, dE)
        wd_active = ufl.inner(self.S_active, dE)
        wd_passive = ufl.inner(self.S_passive, dE)
        wd_comp = ufl.inner(self.S_comp, dE)
        
        f0 = self.fiber_fields['f0']
        s0 = self.fiber_fields['s0'] 
        n0 = self.fiber_fields['n0'] 
        
        def proj(T, v): return ufl.inner(ufl.dot(T, v), v)
        
        wd_fiber = proj(self.S_total, f0) * proj(dE, f0)
        wd_sheet = proj(self.S_total, s0) * proj(dE, s0)
        wd_normal = proj(self.S_total, n0) * proj(dE, n0)
        wd_shear = wd_total - (wd_fiber + wd_sheet + wd_normal)
        wd_passive_fiber = proj(self.S_passive, f0) * proj(dE, f0)

        # 3. Integration
        data = {}
        regions = self._get_regions_to_integrate()
        
        for region_name, cell_tags, region_markers in regions:
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata={"quadrature_degree": 4})
            
            def assemble_region(expr):
                val = 0.0
                for m in region_markers:
                    form = dolfinx.fem.form(expr * dx_sub(int(m)))
                    val += dolfinx.fem.assemble_scalar(form)
                return self.comm.allreduce(val, op=MPI.SUM)

            data[f"work_true_{region_name}"] = assemble_region(wd_total)
            data[f"work_active_{region_name}"] = assemble_region(wd_active)
            data[f"work_passive_{region_name}"] = assemble_region(wd_passive)
            data[f"work_comp_{region_name}"] = assemble_region(wd_comp)
            
            data[f"work_fiber_{region_name}"] = assemble_region(wd_fiber)
            data[f"work_sheet_{region_name}"] = assemble_region(wd_sheet)   
            data[f"work_normal_{region_name}"] = assemble_region(wd_normal) 
            data[f"work_shear_{region_name}"] = assemble_region(wd_shear)   
            data[f"work_passive_fiber_{region_name}"] = assemble_region(wd_passive_fiber)
            
        return data
    
    def _calculate_pressure_proxies(self, model_history, current_state=None):
        """
        Calculate PV-based proxies with Artifact Protection & Unit Scaling.
        """
        proxies = {}
        current_state = current_state or {}
        
        # Unit Conversions
        MMHG_TO_PA = 133.322
        ML_TO_M3 = 1e-6
        
        # 1. Get Inputs (Convert to SI Units)
        p_LV_Pa = (current_state.get("p_LV", 0.0) or 0.0) * MMHG_TO_PA
        p_RV_Pa = (current_state.get("p_RV", 0.0) or 0.0) * MMHG_TO_PA
        
        V_LV_m3 = (current_state.get("V_LV", 0.0) or 0.0) * ML_TO_M3
        V_RV_m3 = (current_state.get("V_RV", 0.0) or 0.0) * ML_TO_M3
        
        # 2. INITIALIZATION CHECK (Prevents the "Inflation Spike")
        if self.V_LV_prev is None:
            self.V_LV_prev = V_LV_m3
            self.V_RV_prev = V_RV_m3
            return {"work_proxy_pv_LV": 0.0, "work_proxy_pv_RV": 0.0}
            
        # 3. Calculate Increments
        dV_LV = V_LV_m3 - self.V_LV_prev
        dV_RV = V_RV_m3 - self.V_RV_prev
        
        # 4. ARTIFACT PROTECTION
        # If dV > 10mL in one step, ignore it (solver reset/glitch)
        if abs(dV_LV) > 1e-5: dV_LV = 0.0
        if abs(dV_RV) > 1e-5: dV_RV = 0.0
        
        # 5. CALCULATE WORK (External Work P*dV in Joules)
        proxies["work_proxy_pv_LV"] = p_LV_Pa * dV_LV
        proxies["work_proxy_pv_RV"] = p_RV_Pa * dV_RV
        
        # Update Previous State
        self.V_LV_prev = V_LV_m3
        self.V_RV_prev = V_RV_m3
        
        return proxies

    def _calculate_pressure_strain_work(self, current_state, mechanics_data):
        """Calculates Pressure-Strain Work Index (PSWI)."""
        data = {}
        # Convert to Pa
        p_LV = (current_state.get("p_LV", 0.0) or 0.0) * 133.322
        p_RV = (current_state.get("p_RV", 0.0) or 0.0) * 133.322
        
        # Get Mean Strains for each Region
        eps_LV = mechanics_data.get("mean_E_ff_LV", 0.0)
        eps_RV = mechanics_data.get("mean_E_ff_RV", 0.0)
        eps_Septum = mechanics_data.get("mean_E_ff_Septum", 0.0)
        
        # Calculate Increments
        dE_LV = eps_LV - self.eps_LV_prev
        dE_RV = eps_RV - self.eps_RV_prev
        dE_Septum = eps_Septum - self.eps_Septum_prev
        
        # --- Apply Regional Volume Scaling ---
        # 1. LV Free Wall Proxy (P_LV * dE_LV * Vol_LV)
        data["work_ps_index_LV"] = (p_LV * dE_LV) * self.region_volumes["LV"]
        
        # 2. RV Free Wall Proxy (P_RV * dE_RV * Vol_RV)
        data["work_ps_index_RV"] = (p_RV * dE_RV) * self.region_volumes["RV"]
        
        # 3. Septum Proxy ((P_LV - P_RV) * dE_Septum * Vol_Septum)
        # The septum works against the trans-septal pressure gradient.
        # Check sign convention: If P_LV > P_RV, and Septum shortens (dE < 0), work should be positive?
        # Standard: Work = Integral( S : dE ). S is tensile > 0. dE is shortening < 0. Work < 0 (Energy consumed/output).
        # Internal Work by Solver is usually negative (Energy Leaving the system).
        # Positive Pressure * Negative Strain -> Negative Work. 
        # So using (P_LV - P_RV) is correct if P_LV is the dominant load.
        data["work_ps_index_Septum"] = ((p_LV - p_RV) * dE_Septum) * self.region_volumes["Septum"]
        
        # Update History
        self.eps_LV_prev = eps_LV
        self.eps_RV_prev = eps_RV
        self.eps_Septum_prev = eps_Septum
        
        return data
    
    def _calculate_robin_work(self):
        """
        Calculates work done BY the boundary springs ON the mesh.
        """
        du = self.problem.u - self._u_prev if hasattr(self, '_u_prev') else self.problem.u
        u_cur = self.problem.u
        
        # Tags for boundary
        tag_epi = self.geometry.markers["EPI"][0]
        tag_base = self.geometry.markers["BASE"][0]
        
        ds_epi = self.geometry.ds(tag_epi)
        ds_base = self.geometry.ds(tag_base)
        
        # Form definition
        term_epi = -self.alpha_epi * ufl.inner(u_cur, du) * ds_epi
        term_base = -self.alpha_base * ufl.inner(u_cur, du) * ds_base
        
        # Assembly (No Try/Except!)
        wd_epi = self.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(term_epi)), op=MPI.SUM)
        wd_base = self.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(term_base)), op=MPI.SUM)
            
        return {"work_robin_epi": wd_epi, "work_robin_base": wd_base}

    def _get_regions_to_integrate(self):
        regions = []
        if self.region_tags is not None:
            regions.append(("LV", self.region_tags, [1])) 
            regions.append(("RV", self.region_tags, [2]))
            regions.append(("Septum", self.region_tags, [3]))
            regions.append(("Whole", self.region_tags, [1, 2, 3, 4]))
        return regions

    def setup_csv_logging(self, file_path):
        if self.rank != 0: return
        self.trace_path = Path(file_path)
        # Reset file
        if self.trace_path.exists():
            try:
                self.trace_path.unlink()
            except Exception:
                pass
        self.trace_headers_written = False

    def store_metrics(self, region_metrics, timestep_idx, t, downsample_factor=1):
        # Always store in memory
        if timestep_idx % downsample_factor == 0:
            self.metrics_history["time"].append(t)
            for key, value in region_metrics.items():
                self.metrics_history[key].append(value)
        
        # Write to CSV if enabled (Rank 0 only)
        if self.rank == 0 and hasattr(self, 'trace_path'):
            # Combine time and metrics for the row
            curr_data = {"step": timestep_idx, "time": t}
            # Add metrics (flatten any numpy scalars)
            for k, v in region_metrics.items():
                if hasattr(v, 'item'):
                     curr_data[k] = v.item()
                else:
                     curr_data[k] = v
            
            if not self.trace_headers_written:
                # Ensure directory exists
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.trace_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=curr_data.keys())
                    writer.writeheader()
                self.trace_headers_written = True
                self.trace_fieldnames = list(curr_data.keys())
            
            with open(self.trace_path, "a", newline="") as f:
                 writer = csv.DictWriter(f, fieldnames=self.trace_fieldnames, extrasaction='ignore')
                 writer.writerow(curr_data)

    def save_metrics(self, output_dir, downsample_factors=[1]):
        if self.rank != 0: return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for factor in downsample_factors:
            downsampled = {k: v[::factor] for k, v in self.metrics_history.items()}
            np.save(output_dir / f"metrics_downsample_{factor}.npy", downsampled, allow_pickle=True)

    def compute_regional_metrics(self, timestep_idx, t, model_history, skip_work_calc=False, current_state=None):
        metrics = {}
        
        # 1. ALWAYS calculate state variables (Stress/Strain exist at t=0)
        # This fixes the KeyError because "mean_S_ff_LV" is now always created
        state_data = self._calculate_state_variables()
        metrics.update(state_data)

        # 2. Work Calculation
        # Work requires a previous state (dt). However, for logging purposes,
        # we MUST ensure the keys exist at t=0, otherwise the CSV writer will
        # permanently drop them if they appear later.
        if self.has_previous_state and not skip_work_calc:
            # A. Mechanics Work
            work_data = self._calculate_incremental_work()
            metrics.update(work_data)
            
            # B. PV Loop Work
            metrics.update(self._calculate_pressure_proxies(model_history, current_state))
            
            # C. PS Work
            metrics.update(self._calculate_pressure_strain_work(current_state, state_data))
            
            # D. Robin Work
            metrics.update(self._calculate_robin_work())
            
            if self.rank == 0:
                w_tot = metrics.get("work_true_LV", 0.0)
                ps_idx = metrics.get("work_ps_index_LV", 0.0)
                print(f"STATS | t={t:.3f} | W_Tot={w_tot:.4e} | PS_Idx={ps_idx:.4e}")

        elif not self.has_previous_state:
             # Initialize all work keys to 0.0 at Step 0 to "reserve" the CSV columns.
             # We perform a dummy call to get the keys, but we don't use the values.
             # Note: This requires temporary state setup or manual key definition.
             # Safer approach: Manually define the expected keys based on regions.
             region_suffixes = [r[0] for r in self._get_regions_to_integrate()]
             
             # Mechanics keys
             prefixes = ["work_true", "work_active", "work_passive", "work_comp", 
                         "work_fiber", "work_sheet", "work_normal", "work_shear", "work_passive_fiber"]
             for r in region_suffixes:
                 for p in prefixes:
                     metrics[f"{p}_{r}"] = 0.0

             # Robin keys
             metrics["work_robin_epi"] = 0.0
             metrics["work_robin_base"] = 0.0
             
             # PV/PS keys
             metrics["work_proxy_pv_LV"] = 0.0
             metrics["work_proxy_pv_RV"] = 0.0
             metrics["work_ps_index_LV"] = 0.0
             metrics["work_ps_index_RV"] = 0.0
             metrics["work_ps_index_Septum"] = 0.0
        
        print(metrics)
        
        return metrics