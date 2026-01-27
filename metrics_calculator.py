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
        
        # Region Tags
        self.region_tags = geo.additional_data.get("markers_mt", None)

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
        """Calculate PV-based proxies."""
        proxies = {}
        current_state = current_state or {}
        
        p_LV_mmHg = current_state.get("p_LV", 0.0)
        p_RV_mmHg = current_state.get("p_RV", 0.0)
        
        p_LV_Pa = p_LV_mmHg * 133.322
        p_RV_Pa = p_RV_mmHg * 133.322
        
        V_LV_m3 = current_state.get("V_LV", 0.0) * 1e-6
        V_RV_m3 = current_state.get("V_RV", 0.0) * 1e-6
        
        if self.V_LV_prev is None:
            self.V_LV_prev = V_LV_m3
            self.V_RV_prev = V_RV_m3
            return {k: 0.0 for k in ["work_proxy_pv_LV", "work_proxy_pv_RV"]}
            
        dV_LV = V_LV_m3 - self.V_LV_prev
        dV_RV = V_RV_m3 - self.V_RV_prev
        
        proxies["work_proxy_pv_LV"] = p_LV_Pa * dV_LV
        proxies["work_proxy_pv_RV"] = p_RV_Pa * dV_RV
        
        self.V_LV_prev = V_LV_m3
        self.V_RV_prev = V_RV_m3
        return proxies

    def _calculate_pressure_strain_work(self, current_state, mechanics_data):
        """Calculates Pressure-Strain Work Index (PSWI)."""
        data = {}
        p_LV = current_state.get("p_LV", 0.0) * 133.322
        p_RV = current_state.get("p_RV", 0.0) * 133.322
        
        eps_LV = mechanics_data.get("mean_E_ff_LV", 0.0)
        eps_RV = mechanics_data.get("mean_E_ff_RV", 0.0)
        
        dE_LV = eps_LV - self.eps_LV_prev
        dE_RV = eps_RV - self.eps_RV_prev
        
        data["work_ps_index_LV"] = p_LV * dE_LV
        data["work_ps_index_RV"] = p_RV * dE_RV
        
        self.eps_LV_prev = eps_LV
        self.eps_RV_prev = eps_RV
        return data
    
    def _calculate_robin_work(self):
        """
        Calculates work done BY the boundary springs ON the mesh.
        UNSILENCED: This will crash if alpha values are not found or variables mismatch.
        """
        du = self.problem.u - self._u_prev if hasattr(self, '_u_prev') else self.problem.u
        u_cur = self.problem.u
        
        # Tags for boundary
        tag_epi = self.geometry.markers["EPI"][0]
        tag_base = self.geometry.markers["BASE"][0]
        
        ds_epi = self.geometry.ds(tag_epi)
        ds_base = self.geometry.ds(tag_base)
        
        # WARNING: Hardcoded default stiffnesses. 
        # Ideal: Pass these in __init__ from main.py args.
        alpha_epi_val = 1e5 
        alpha_base_val = 1e6 
        
        # Form definition
        term_epi = -alpha_epi_val * ufl.inner(u_cur, du) * ds_epi
        term_base = -alpha_base_val * ufl.inner(u_cur, du) * ds_base
        
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

    def store_metrics(self, region_metrics, timestep_idx, t, downsample_factor=1):
        if timestep_idx % downsample_factor != 0: return
        self.metrics_history["time"].append(t)
        for key, value in region_metrics.items():
            self.metrics_history[key].append(value)

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

        # 2. CONDITIONALLY calculate work (requires dt)
        if not skip_work_calc and self.has_previous_state:
            # Mechanics Work
            work_data = self._calculate_incremental_work()
            metrics.update(work_data)
            
            # PV Loop Work
            metrics.update(self._calculate_pressure_proxies(model_history, current_state))
            
            # PS Work (Pass state_data so we can use current Strain!)
            metrics.update(self._calculate_pressure_strain_work(current_state, state_data))
            
            # Robin Work
            metrics.update(self._calculate_robin_work())
            
            if self.rank == 0:
                w_tot = metrics.get("work_true_LV", 0.0)
                ps_idx = metrics.get("work_ps_index_LV", 0.0)
                print(f"STATS | t={t:.3f} | W_Tot={w_tot:.4e} | PS_Idx={ps_idx:.4e}")
        
        return metrics