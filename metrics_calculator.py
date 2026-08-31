"""
metrics_calculator.py — FEM-level metric integration for the offline postprocessor.

As of the 2026-04-13 refactor this module is organized into three buckets.
Only the first is computed by default.

    A. BOUNDARY + LOOPS (ALWAYS ON)
       Boundary work via Nanson's formula on the LV/RV cavity surfaces,
       Robin epi/base work, PV/PS/SS loop time-series (mean_E_ff, mean_S_ff,
       mean_E_ll, mean_S_ll per LDRB region), energy balance diagnostics.
       compute_per_cell.py does NOT produce these, so they stay here.

    B. REGIONAL INTERNAL WORK  (gated by enable_regional_internal, default OFF)
       Per-region S:dE totals and their active/passive/comp breakdown with
       ff/ss/nn/cross directional decomposition. compute_per_cell.py is now
       the canonical source for these quantities.

    C. RESEARCH METRICS  (gated by enable_research, default OFF — reserved)
       Fiber efficiency, dyssynchrony, work redistribution, and other
       non-canonical research diagnostics. These currently live downstream
       in compare_cases.py / eval_proxies.py. The flag is a stable entry
       point for future migrations: new research metrics added to this
       file should be gated by self.enable_research.

In addition, enable_decomp gates the directional decomposition of mean
state variables (mean_S/E_ss/nn and the full Cauchy-stress decomposition)
which is distinct from regional internal work but equally expensive.

The three flags are independent; enable_decomp can be on while
enable_regional_internal is off (or vice-versa), though the production
path is all three off.
"""

import numpy as np
import csv
from pathlib import Path
from collections import defaultdict
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import basix.ufl
import pulse

class MetricsCalculator:
    def __init__(self, geometry, geo, fiber_field_map, problem, comm, cardiac_model, metrics_space_type=("DG", 1), alpha_epi=1e5, alpha_base=1e6, hydro_pressure=None, one_sided_robin=False, aha_tags=None, enable_decomp=False, enable_research=False, enable_regional_internal=False):
        self.geometry = geometry
        self.geo = geo
        self.fiber_fields = fiber_field_map
        self.cardiac_model = cardiac_model

        # Refactor 2026-04-13: canonical regional work lives in compute_per_cell.py.
        # These flags gate the legacy regional + decomposition paths. Defaults are
        # OFF; the caller (postprocess_metrics.py) re-enables them via --no-skip-*.
        self.enable_decomp = enable_decomp
        self.enable_research = enable_research
        self.enable_regional_internal = enable_regional_internal
        
        # Register pressure for stress calculations if incompressible
        if hydro_pressure is not None:
             if isinstance(self.cardiac_model.compressibility, pulse.compressibility.Incompressible):
                 self.cardiac_model.compressibility.register(hydro_pressure)
                 if comm.rank == 0:
                     print("MetricsCalculator: Registered hydrostatic pressure for stress calculation.")
        self.problem = problem
        self.comm = comm
        self.rank = comm.rank
        self.alpha_epi = alpha_epi
        self.alpha_base = alpha_base
        self.one_sided_robin = one_sided_robin
        self.mesh = geometry.mesh
        
        # --- 1. Define Function Spaces ---
        element_type = metrics_space_type[0]
        degree = metrics_space_type[1]
        self._state_space_is_quadrature = element_type == "Quadrature"

        # Determine integration quadrature degree
        # For Quadrature elements, we MUST integrate with the same degree to match points.
        # For DG/Lagrange, we default to 4 (as per original code), or higher if needed.
        if self._state_space_is_quadrature:
            self.quadrature_degree = degree
            if self.rank == 0:
                 print(f"MetricsCalculator: Using Quadrature Element (Deg {degree}). Quadrature Degree for Integration: {self.quadrature_degree}")
            # Use basix.ufl to create Quadrature elements explicitly
            el_scalar = basix.ufl.quadrature_element(self.mesh.topology.cell_name(), degree=degree)
            self.W_scalar = dolfinx.fem.functionspace(self.mesh, el_scalar)
            
            el_tensor = basix.ufl.quadrature_element(self.mesh.topology.cell_name(), value_shape=(3,3), degree=degree)
            self.W_tensor = dolfinx.fem.functionspace(self.mesh, el_tensor)
        else:
            # Fibre fields are stored in degree-6 quadrature space. Keep the
            # integration rule at least degree 6 even when stress/strain state
            # storage is deliberately degraded to DG0/DG1 for sensitivity tests.
            self.quadrature_degree = max(6, degree + 2)
            
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

        self.S_prev = dolfinx.fem.Function(self.W_tensor, name="S_prev")
        self.S_prev.x.array[:] = 0.0

        self.S_active_prev = dolfinx.fem.Function(self.W_tensor, name="S_active_prev")
        self.S_active_prev.x.array[:] = 0.0
        self.S_passive_prev = dolfinx.fem.Function(self.W_tensor, name="S_passive_prev")
        self.S_passive_prev.x.array[:] = 0.0
        self.S_comp_prev = dolfinx.fem.Function(self.W_tensor, name="S_comp_prev")
        self.S_comp_prev.x.array[:] = 0.0

        # --- Cauchy Stress Functions (Current Configuration) ---
        self.sigma_total = dolfinx.fem.Function(self.W_tensor, name="sigma_total")
        self.sigma_active = dolfinx.fem.Function(self.W_tensor, name="sigma_active")
        self.sigma_passive = dolfinx.fem.Function(self.W_tensor, name="sigma_passive")
        self.sigma_comp = dolfinx.fem.Function(self.W_tensor, name="sigma_comp")
        self._state_projection_problems = {}

        # --- 3. Setup UFL Expressions ---
        self._setup_expressions()

        # Will be populated after region_tags is set (in __init__ after marker diagnostic)
        self._compiled_forms = None

        # History and State flags
        self.metrics_history = defaultdict(list)
        self.has_previous_state = False
        self.V_LV_prev = None
        self.V_RV_prev = None
        
        # Strain History for PS Loops — dict-based for extensibility to AHA regions
        self._eps_prev = {}  # keyed by "{strain_component}_{region}", e.g. "E_ff_LV"

        # Pressure History for Trapezoidal Boundary Work
        self.p_LV_prev = 0.0
        self.p_RV_prev = 0.0
        self._p_initialized = False
        
        # Region Tags
        self.region_tags = geo.additional_data.get("markers_mt", None)

        # AHA sub-region tags (Basal/Mid split of LV/RV/Septum)
        self.aha_tags = aha_tags

        # Tau-based LV/RV split — TWO independent definitions computed for
        # the simplification-cascade pedagogy and as a robustness check on
        # the region split definition. Every wall cell is classified into
        # LV territory or RV territory under each definition; no cell is
        # excluded. Each territory is loaded by an unambiguous cavity
        # pressure (P_LV or P_RV), which is the point of the split.
        #
        # (a) Euclidean:  tau_eu = d_LV / (d_LV + d_RV),
        #                 LV iff tau_eu < 0.5. Deterministic geometric
        #                 definition — "which endocardium is this cell
        #                 closer to?" — uses nearest-vertex Euclidean
        #                 distances to the two endocardial surfaces.
        #
        # (b) Laplace:    tau_lap solves ∇²u = 0 with u=0 on LV endo,
        #                 u=1 on RV endo. LV iff tau_lap < 0.5. Smooth
        #                 harmonic interpolation between the two cavities;
        #                 the 0.5 isosurface is the electrostatic
        #                 equipotential between them. Matches the LDRB
        #                 algorithm's own internal tau field.
        #
        # Tag values: 11 = LV_tau_* (tau < 0.5), 12 = RV_tau_* (tau >= 0.5).
        self.tau_tags_eu = self._build_tau_tags_euclid(geo)
        self.tau_tags_lap = self._build_tau_tags_laplace(geo)

        # --- 4. Calculate Regional Wall Volumes for Unit Scaling ---
        # These scale the PS indices, so a wrong volume is a wrong work density
        # (~4 orders of magnitude if it defaults to 1.0 m³ instead of ~1e-4).
        # Failures here must propagate rather than fall back to a placeholder.
        self.region_volumes = {}
        # Helper to integrate volume for a specific tag
        def get_vol(tags):
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.region_tags, metadata={"quadrature_degree": self.quadrature_degree})
            val = 0.0
            for t in tags:
                 val += dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(self.mesh, 1.0) * dx_sub(int(t))))
            return self.comm.allreduce(val, op=MPI.SUM)

        if self.region_tags:
            self.region_volumes["LV"] = get_vol([1])      # LV Free Wall
            self.region_volumes["RV"] = get_vol([2])      # RV Free Wall
            self.region_volumes["Septum"] = get_vol([3])  # Septum
            self.region_volumes["Whole"] = get_vol([1, 2, 3, 4]) # Whole Mesh

        def get_tau_vol(tau_mt, tag_val):
            if tau_mt is None:
                return 0.0
            dx_tau = ufl.Measure("dx", domain=self.mesh,
                                  subdomain_data=tau_mt,
                                  metadata={"quadrature_degree": self.quadrature_degree})
            val = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(dolfinx.fem.Constant(self.mesh, 1.0)
                                  * dx_tau(int(tag_val))))
            return self.comm.allreduce(val, op=MPI.SUM)

        if self.tau_tags_eu is not None:
            self.region_volumes["LV_tau_eu"] = get_tau_vol(self.tau_tags_eu, 11)
            self.region_volumes["RV_tau_eu"] = get_tau_vol(self.tau_tags_eu, 12)
        if self.tau_tags_lap is not None:
            self.region_volumes["LV_tau_lap"] = get_tau_vol(self.tau_tags_lap, 11)
            self.region_volumes["RV_tau_lap"] = get_tau_vol(self.tau_tags_lap, 12)

        # AHA sub-region volumes
        if self.aha_tags is not None:
            def get_aha_vol(tags):
                dx_aha = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.aha_tags, metadata={"quadrature_degree": self.quadrature_degree})
                val = 0.0
                for t in tags:
                    val += dolfinx.fem.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(self.mesh, 1.0) * dx_aha(int(t))))
                return self.comm.allreduce(val, op=MPI.SUM)

            aha_region_map = {
                "Basal_LV": [1], "Basal_RV": [2], "Basal_Septum": [3],
                "Mid_LV": [4], "Mid_RV": [5], "Mid_Septum": [6],
                "Apical": [0],
            }
            for name, tags in aha_region_map.items():
                self.region_volumes[name] = get_aha_vol(tags)


        # --- Marker Diagnostic (MPI-global counts) ---
        if self.region_tags is not None:
            tag_values = self.region_tags.values
            # Count per marker locally, then allreduce
            max_marker = 4
            local_counts = np.zeros(max_marker + 1, dtype=np.int64)
            for v in tag_values:
                if 0 <= int(v) <= max_marker:
                    local_counts[int(v)] += 1
            global_counts = np.zeros_like(local_counts)
            self.comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
            total_global = int(global_counts.sum())

            if self.rank == 0:
                print(f"\n{'='*60}")
                print(f"  METRICS CALCULATOR — Region Marker Diagnostic")
                print(f"{'='*60}")
                print(f"  Total tagged cells (global): {total_global}")
                marker_labels = {0: "UNASSIGNED", 1: "LV FreeWall", 2: "RV FreeWall", 3: "Septum", 4: "Other"}
                for i in range(max_marker + 1):
                    if global_counts[i] > 0:
                        label = marker_labels.get(i, f"Unknown({i})")
                        print(f"    Marker {i:2d} ({label:>14s}): {global_counts[i]:6d} cells")
                if global_counts[0] > 0:
                    print(f"  WARNING: {global_counts[0]} cells have marker 0 (unassigned)!")
                    print(f"           These cells contribute to solver S:dE but NOT to metrics 'Whole' region.")
                else:
                    print(f"  OK: All cells assigned to regions 1/2/3.")
                for rname, rvol in self.region_volumes.items():
                    print(f"  Volume {rname:>14s}: {rvol:.4e} m^3")
                print(f"{'='*60}\n")

        # --- 5. Pre-compile all integration forms (one-time JIT cost) ---
        self._precompile_forms()
        if self.rank == 0:
            print("MetricsCalculator: All forms pre-compiled.")

    def _precompile_forms(self):
        """Pre-compile all dolfinx.fem.form objects used in the hot loop.

        Since all UFL expressions reference dolfinx.fem.Function objects,
        updating .x.array on those functions automatically updates form values.
        We only need to call assemble_scalar(form) each step — no re-compilation.
        """
        self._compiled_forms = {}
        regions = self._get_regions_to_integrate()

        # --- Fiber directions and push-forward vectors ---
        u = self.problem.u
        F = ufl.variable(ufl.grad(u) + ufl.Identity(3))

        f0 = self.fiber_fields['f0']
        s0 = self.fiber_fields['s0']
        n0 = self.fiber_fields['n0']
        l0_raw = self.fiber_fields.get('l0', None)

        if l0_raw is not None:
            l0_norm = ufl.sqrt(ufl.inner(l0_raw, l0_raw))
            l0 = l0_raw / l0_norm
        else:
            l0 = None

        def push_vec(v0):
            v_cur = F * v0
            return v_cur / ufl.sqrt(ufl.inner(v_cur, v_cur))

        f_cur = push_vec(f0)
        s_cur = push_vec(s0)
        n_cur = push_vec(n0)
        l_cur = push_vec(l0) if l0 is not None else None

        def proj(T, v): return ufl.inner(ufl.dot(T, v), v)
        mag = lambda T: ufl.sqrt(ufl.inner(T, T))

        # --- State variable forms (per region) ---
        one = dolfinx.fem.Constant(self.mesh, 1.0)

        for region_name, cell_tags, region_markers in regions:
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags,
                                 metadata={"quadrature_degree": self.quadrature_degree})

            def compile_regional(expr, name):
                """Compile form for each marker in the region, store as list."""
                forms = []
                for m in region_markers:
                    forms.append(dolfinx.fem.form(expr * dx_sub(int(m))))
                self._compiled_forms[f"{name}_{region_name}"] = forms

            # Volume — always needed (PS loop denominators, boundary checks)
            compile_regional(one, "vol")

            # PS-loop essentials — ALWAYS compiled (boundary/loops bucket).
            # mean_E_ff / mean_S_ff / (mean_E_ll / mean_S_ll when l0 exists)
            # drive the pressure-strain loop time-series.
            compile_regional(proj(self.S_total, f0), "S_ff")
            compile_regional(proj(self.E_cur, f0), "E_ff")
            if l0 is not None:
                compile_regional(proj(self.S_total, l0), "S_ll")
                compile_regional(proj(self.E_cur, l0), "E_ll")

            # Directional decomposition of state variables — gated.
            # Everything below this point is ss/nn projections and Cauchy
            # stress decomposition used for research diagnostics, not for
            # any canonical energy balance or clinical proxy.
            if self.enable_decomp:
                compile_regional(proj(self.S_total, s0), "S_ss")
                compile_regional(proj(self.S_total, n0), "S_nn")
                compile_regional(proj(self.E_cur, s0), "E_ss")
                compile_regional(proj(self.E_cur, n0), "E_nn")

                # Cauchy stresses - total
                compile_regional(proj(self.sigma_total, f_cur), "sigma_ff")
                compile_regional(proj(self.sigma_total, s_cur), "sigma_ss")
                compile_regional(proj(self.sigma_total, n_cur), "sigma_nn")
                if l_cur is not None:
                    compile_regional(proj(self.sigma_total, l_cur), "sigma_ll")

                # Cauchy - active
                compile_regional(proj(self.sigma_active, f_cur), "sigma_ff_active")
                compile_regional(proj(self.sigma_active, s_cur), "sigma_ss_active")
                compile_regional(proj(self.sigma_active, n_cur), "sigma_nn_active")

                # Cauchy - passive
                compile_regional(proj(self.sigma_passive, f_cur), "sigma_ff_passive")
                compile_regional(proj(self.sigma_passive, s_cur), "sigma_ss_passive")
                compile_regional(proj(self.sigma_passive, n_cur), "sigma_nn_passive")

                # Cauchy - comp
                compile_regional(proj(self.sigma_comp, f_cur), "sigma_ff_comp")
                compile_regional(proj(self.sigma_comp, s_cur), "sigma_ss_comp")
                compile_regional(proj(self.sigma_comp, n_cur), "sigma_nn_comp")

                # Cauchy magnitudes
                compile_regional(mag(self.sigma_total), "sigma_mag")
                compile_regional(mag(self.sigma_active), "sigma_mag_active")
                compile_regional(mag(self.sigma_passive), "sigma_mag_passive")
                compile_regional(mag(self.sigma_comp), "sigma_mag_comp")

        # --- Work forms (per region) — gated by enable_regional_internal ---
        # compute_per_cell.py is now the canonical source for these integrals.
        # Skipping both form compilation AND per-timestep assembly saves
        # 30+ form JITs and the matching assemble_scalar calls.
        if self.enable_regional_internal:
            dE = self.E_cur - self.E_prev
            wd_total = 0.5 * ufl.inner(self.S_tot_ufl + self.S_prev, dE)
            wd_active = 0.5 * ufl.inner(self.S_act_ufl + self.S_active_prev, dE)
            wd_passive = 0.5 * ufl.inner(self.S_pas_ufl + self.S_passive_prev, dE)
            wd_comp = 0.5 * ufl.inner(self.S_cmp_ufl + self.S_comp_prev, dE)

            wd_fiber = 0.5 * (proj(self.S_tot_ufl, f0) + proj(self.S_prev, f0)) * proj(dE, f0)
            wd_sheet = 0.5 * (proj(self.S_tot_ufl, s0) + proj(self.S_prev, s0)) * proj(dE, s0)
            wd_normal = 0.5 * (proj(self.S_tot_ufl, n0) + proj(self.S_prev, n0)) * proj(dE, n0)
            wd_shear = wd_total - (wd_fiber + wd_sheet + wd_normal)
            wd_passive_fiber = 0.5 * (proj(self.S_pas_ufl, f0) + proj(self.S_passive_prev, f0)) * proj(dE, f0)

            for region_name, cell_tags, region_markers in regions:
                dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags,
                                     metadata={"quadrature_degree": self.quadrature_degree})

                def compile_regional_work(expr, name):
                    forms = []
                    for m in region_markers:
                        forms.append(dolfinx.fem.form(expr * dx_sub(int(m))))
                    self._compiled_forms[f"{name}_{region_name}"] = forms

                compile_regional_work(wd_total, "wd_total")
                compile_regional_work(wd_active, "wd_active")
                compile_regional_work(wd_passive, "wd_passive")
                compile_regional_work(wd_comp, "wd_comp")
                compile_regional_work(wd_fiber, "wd_fiber")
                compile_regional_work(wd_sheet, "wd_sheet")
                compile_regional_work(wd_normal, "wd_normal")
                compile_regional_work(wd_shear, "wd_shear")
                compile_regional_work(wd_passive_fiber, "wd_passive_fiber")

        # --- Robin work forms ---
        N = ufl.FacetNormal(self.mesh)
        I = ufl.Identity(3)
        F_robin = ufl.grad(u) + I
        J_robin = ufl.det(F_robin)
        cof = J_robin * ufl.inv(F_robin).T
        cofN = ufl.dot(cof, N)
        cofnorm = ufl.sqrt(ufl.inner(cofN, cofN))
        NN = cofN / cofnorm

        # _u_prev needed for robin — create it now
        if not hasattr(self, '_u_prev'):
            self._u_prev = dolfinx.fem.Function(self.problem.u.function_space)

        u_avg = 0.5 * (u + self._u_prev)
        du = u - self._u_prev
        u_n = ufl.dot(u_avg, NN)
        du_n = ufl.dot(du, NN)

        # One-sided Robin: only resists outward displacement (u·NN > 0)
        if self.one_sided_robin:
            u_n_cur = ufl.dot(u, NN)
            u_n_prev = ufl.dot(self._u_prev, NN)
            # Use conditional on current normal displacement to match solver
            u_n = ufl.conditional(ufl.gt(u_n_cur, 0), u_n, ufl.as_ufl(0.0))
            du_n = ufl.conditional(ufl.gt(u_n_cur, 0), du_n, ufl.as_ufl(0.0))

        tag_epi = self.geometry.markers["EPI"][0]
        tag_base = self.geometry.markers["BASE"][0]
        ds_robin = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.geometry.facet_tags)

        self._compiled_forms["robin_epi"] = dolfinx.fem.form(
            -self.alpha_epi * u_n * du_n * cofnorm * ds_robin(tag_epi))
        self._compiled_forms["robin_base"] = dolfinx.fem.form(
            -self.alpha_base * u_n * du_n * cofnorm * ds_robin(tag_base))

        # --- Boundary work exact forms ---
        X = ufl.SpatialCoordinate(self.mesh)
        F_cur_bnd = ufl.grad(u) + I
        J_cur_bnd = ufl.det(F_cur_bnd)
        vol_form_cur = (-1.0 / 3.0) * J_cur_bnd * ufl.dot(X + u, ufl.dot(ufl.inv(F_cur_bnd).T, N))

        F_prev_bnd = ufl.grad(self._u_prev) + I
        J_prev_bnd = ufl.det(F_prev_bnd)
        vol_form_prev = (-1.0 / 3.0) * J_prev_bnd * ufl.dot(X + self._u_prev, ufl.dot(ufl.inv(F_prev_bnd).T, N))

        lv_marker_name = "LV" if "LV" in self.geometry.markers else "ENDO_LV"
        rv_marker_name = "RV" if "RV" in self.geometry.markers else "ENDO_RV"
        tag_lv = self.geometry.markers[lv_marker_name][0]
        tag_rv = self.geometry.markers[rv_marker_name][0]
        ds_bnd = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.geometry.facet_tags)

        self._compiled_forms["vol_cur_LV"] = dolfinx.fem.form(vol_form_cur * ds_bnd(tag_lv))
        self._compiled_forms["vol_prev_LV"] = dolfinx.fem.form(vol_form_prev * ds_bnd(tag_lv))
        self._compiled_forms["vol_cur_RV"] = dolfinx.fem.form(vol_form_cur * ds_bnd(tag_rv))
        self._compiled_forms["vol_prev_RV"] = dolfinx.fem.form(vol_form_prev * ds_bnd(tag_rv))

    def _assemble_compiled(self, key):
        """Assemble a pre-compiled form (or list of forms for multi-marker regions)."""
        forms = self._compiled_forms[key]
        if isinstance(forms, list):
            val = 0.0
            for f in forms:
                val += dolfinx.fem.assemble_scalar(f)
            return self.comm.allreduce(val, op=MPI.SUM)
        else:
            return self.comm.allreduce(dolfinx.fem.assemble_scalar(forms), op=MPI.SUM)

    def _make_state_projection_problem(self, expr, target, name, dx_proj):
        """Build an L2 projection into the configured state-storage space.

        Quadrature coefficients cannot generally be evaluated at DG interpolation
        points. For DG0/DG1 sensitivity runs, project the stress expression into
        the storage space instead of trying point interpolation.
        """
        trial = ufl.TrialFunction(self.W_tensor)
        test = ufl.TestFunction(self.W_tensor)
        a = ufl.inner(trial, test) * dx_proj
        L = ufl.inner(expr, test) * dx_proj
        return dolfinx.fem.petsc.LinearProblem(
            a,
            L,
            u=target,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix=f"metrics_proj_{id(self)}_{name}_",
        )

    def _store_state_by_projection(self, name):
        problem = self._state_projection_problems[name]
        fn = problem.solve()
        fn.x.scatter_forward()

    def _setup_expressions(self):
        u = self.problem.u
        I = ufl.Identity(3)
        F = ufl.variable(ufl.grad(u) + I)
        C = ufl.variable(F.T * F) 
        E = 0.5 * (C - I) # Green-Lagrange Strain Tensor
        
        # --- SAVE THESE AS SELF ---
        # These are the exact mathematical definitions
        self.S_tot_ufl = self.cardiac_model.S(C) # Total 2nd Piola-Kirchhoff Stress
        self.S_act_ufl = self.cardiac_model.active.S(C, dev=True) # Active component (use deviatoric if compressible)
        self.S_pas_ufl = self.cardiac_model.material.S(C, dev=True) # use c dev instead
        self.S_cmp_ufl = self.cardiac_model.compressibility.S(C)

        points = self.W_tensor.element.interpolation_points
        self.expr_E = dolfinx.fem.Expression(E, points)

        dx_proj = ufl.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": self.quadrature_degree})
        if self._state_space_is_quadrature:
            self.expr_S_total = dolfinx.fem.Expression(self.S_tot_ufl, points)
            self.expr_S_active = dolfinx.fem.Expression(self.S_act_ufl, points)
            self.expr_S_passive = dolfinx.fem.Expression(self.S_pas_ufl, points)
            self.expr_S_comp = dolfinx.fem.Expression(self.S_cmp_ufl, points)
        else:
            self._state_projection_problems["S_total"] = self._make_state_projection_problem(
                self.S_tot_ufl, self.S_total, "S_total", dx_proj
            )
            if self.enable_regional_internal:
                self._state_projection_problems["S_active"] = self._make_state_projection_problem(
                    self.S_act_ufl, self.S_active, "S_active", dx_proj
                )
                self._state_projection_problems["S_passive"] = self._make_state_projection_problem(
                    self.S_pas_ufl, self.S_passive, "S_passive", dx_proj
                )
                self._state_projection_problems["S_comp"] = self._make_state_projection_problem(
                    self.S_cmp_ufl, self.S_comp, "S_comp", dx_proj
                )

        # --- Cauchy Stress Expressions (Push Forward) ---
        J = ufl.det(F)
        def push_forward(S):
            return (1.0 / J) * F * S * F.T

        sigma_tot_ufl = push_forward(self.S_tot_ufl)
        sigma_act_ufl = push_forward(self.S_act_ufl)
        sigma_pas_ufl = push_forward(self.S_pas_ufl)
        sigma_cmp_ufl = push_forward(self.S_cmp_ufl)

        if self._state_space_is_quadrature:
            self.expr_sigma_total = dolfinx.fem.Expression(sigma_tot_ufl, points)
            self.expr_sigma_active = dolfinx.fem.Expression(sigma_act_ufl, points)
            self.expr_sigma_passive = dolfinx.fem.Expression(sigma_pas_ufl, points)
            self.expr_sigma_comp = dolfinx.fem.Expression(sigma_cmp_ufl, points)
        elif self.enable_decomp:
            self._state_projection_problems["sigma_total"] = self._make_state_projection_problem(
                sigma_tot_ufl, self.sigma_total, "sigma_total", dx_proj
            )
            self._state_projection_problems["sigma_active"] = self._make_state_projection_problem(
                sigma_act_ufl, self.sigma_active, "sigma_active", dx_proj
            )
            self._state_projection_problems["sigma_passive"] = self._make_state_projection_problem(
                sigma_pas_ufl, self.sigma_passive, "sigma_passive", dx_proj
            )
            self._state_projection_problems["sigma_comp"] = self._make_state_projection_problem(
                sigma_cmp_ufl, self.sigma_comp, "sigma_comp", dx_proj
            )

    def update_state(self):
        """Called at the END of a timestep to shift Current -> Prev."""
        self.E_prev.x.array[:] = self.E_cur.x.array[:]
        self.S_prev.x.array[:] = self.S_total.x.array[:]
        
        self.S_active_prev.x.array[:] = self.S_active.x.array[:]
        self.S_passive_prev.x.array[:] = self.S_passive.x.array[:]
        self.S_comp_prev.x.array[:] = self.S_comp.x.array[:]

        # Track displacement for boundary work
        if not hasattr(self, '_u_prev'):
            self._u_prev = dolfinx.fem.Function(self.problem.u.function_space)
        self._u_prev.x.array[:] = self.problem.u.x.array[:]
        
        self.has_previous_state = True
        
    def _calculate_state_variables(self):
        """
        Calculates absolute state variables (Stress, Strain) at the current time.
        Uses pre-compiled forms for fast assembly.
        """
        # --- A. Interpolate Physics ---
        # E_cur and S_total are needed by every path (PS loops, boundary work
        # via shared _u_prev, regional work).
        self.E_cur.interpolate(self.expr_E)
        if self._state_space_is_quadrature:
            self.S_total.interpolate(self.expr_S_total)
        else:
            self._store_state_by_projection("S_total")

        # S_active / S_passive / S_comp are needed by _calculate_incremental_work
        # (regional internal work). Cauchy stresses are only needed by the
        # directional decomposition of state variables.
        if self.enable_regional_internal:
            if self._state_space_is_quadrature:
                self.S_active.interpolate(self.expr_S_active)
                self.S_passive.interpolate(self.expr_S_passive)
                self.S_comp.interpolate(self.expr_S_comp)
            else:
                self._store_state_by_projection("S_active")
                self._store_state_by_projection("S_passive")
                self._store_state_by_projection("S_comp")

        if self.enable_decomp:
            # Cauchy push-forwards: only used by mean_sigma_* assemble below.
            # Interpolating these tensor functions is expensive; skip entirely
            # when the decomp forms do not exist.
            if self._state_space_is_quadrature:
                self.sigma_total.interpolate(self.expr_sigma_total)
                self.sigma_active.interpolate(self.expr_sigma_active)
                self.sigma_passive.interpolate(self.expr_sigma_passive)
                self.sigma_comp.interpolate(self.expr_sigma_comp)
            else:
                self._store_state_by_projection("sigma_total")
                self._store_state_by_projection("sigma_active")
                self._store_state_by_projection("sigma_passive")
                self._store_state_by_projection("sigma_comp")

        # --- B. Integration using pre-compiled forms ---
        data = {}

        # Ta is a uniform scalar Constant under the FrankStarlingActiveStress
        # model, so Variable.value.value is the scalar. A regional (Function)
        # activation has no .value; suppress the key in that case rather than
        # reporting 0.0, which would read as a real "no activation" measurement.
        activation = getattr(self.cardiac_model.active, "activation", None)
        Ta_constant = getattr(getattr(activation, "value", None), "value", None)
        if Ta_constant is not None:
            data["debug_Ta_internal_max"] = float(Ta_constant)
        # debug_S_active_max reads the S_active tensor function, which is only
        # interpolated when enable_regional_internal is on. In slim mode the
        # array keeps its initial (zero) value, so suppress the key entirely
        # to avoid reporting a stale 0.0.
        if self.enable_regional_internal:
            data["debug_S_active_max"] = np.max(self.S_active.x.array)

        l0_available = self.fiber_fields.get('l0', None) is not None
        regions = self._get_regions_to_integrate()

        for region_name, _, _ in regions:
            vol = self._assemble_compiled(f"vol_{region_name}")

            if vol > 1e-12:
                # PS-loop essentials — always computed.
                data[f"mean_S_ff_{region_name}"] = self._assemble_compiled(f"S_ff_{region_name}") / vol
                data[f"mean_E_ff_{region_name}"] = self._assemble_compiled(f"E_ff_{region_name}") / vol
                if l0_available:
                    data[f"mean_S_ll_{region_name}"] = self._assemble_compiled(f"S_ll_{region_name}") / vol
                    data[f"mean_E_ll_{region_name}"] = self._assemble_compiled(f"E_ll_{region_name}") / vol

                # Directional decomposition — gated. Skip assembly AND
                # dict entries so consumers that rely on presence know the
                # flag is off.
                if self.enable_decomp:
                    data[f"mean_S_ss_{region_name}"] = self._assemble_compiled(f"S_ss_{region_name}") / vol
                    data[f"mean_S_nn_{region_name}"] = self._assemble_compiled(f"S_nn_{region_name}") / vol
                    data[f"mean_E_ss_{region_name}"] = self._assemble_compiled(f"E_ss_{region_name}") / vol
                    data[f"mean_E_nn_{region_name}"] = self._assemble_compiled(f"E_nn_{region_name}") / vol

                    data[f"mean_sigma_ff_{region_name}"] = self._assemble_compiled(f"sigma_ff_{region_name}") / vol
                    data[f"mean_sigma_ss_{region_name}"] = self._assemble_compiled(f"sigma_ss_{region_name}") / vol
                    data[f"mean_sigma_nn_{region_name}"] = self._assemble_compiled(f"sigma_nn_{region_name}") / vol
                    if l0_available:
                        data[f"mean_sigma_ll_{region_name}"] = self._assemble_compiled(f"sigma_ll_{region_name}") / vol

                    data[f"mean_sigma_ff_active_{region_name}"] = self._assemble_compiled(f"sigma_ff_active_{region_name}") / vol
                    data[f"mean_sigma_ss_active_{region_name}"] = self._assemble_compiled(f"sigma_ss_active_{region_name}") / vol
                    data[f"mean_sigma_nn_active_{region_name}"] = self._assemble_compiled(f"sigma_nn_active_{region_name}") / vol

                    data[f"mean_sigma_ff_passive_{region_name}"] = self._assemble_compiled(f"sigma_ff_passive_{region_name}") / vol
                    data[f"mean_sigma_ss_passive_{region_name}"] = self._assemble_compiled(f"sigma_ss_passive_{region_name}") / vol
                    data[f"mean_sigma_nn_passive_{region_name}"] = self._assemble_compiled(f"sigma_nn_passive_{region_name}") / vol

                    data[f"mean_sigma_ff_comp_{region_name}"] = self._assemble_compiled(f"sigma_ff_comp_{region_name}") / vol
                    data[f"mean_sigma_ss_comp_{region_name}"] = self._assemble_compiled(f"sigma_ss_comp_{region_name}") / vol
                    data[f"mean_sigma_nn_comp_{region_name}"] = self._assemble_compiled(f"sigma_nn_comp_{region_name}") / vol

                    data[f"mean_sigma_mag_{region_name}"] = self._assemble_compiled(f"sigma_mag_{region_name}") / vol
                    data[f"mean_sigma_mag_active_{region_name}"] = self._assemble_compiled(f"sigma_mag_active_{region_name}") / vol
                    data[f"mean_sigma_mag_passive_{region_name}"] = self._assemble_compiled(f"sigma_mag_passive_{region_name}") / vol
                    data[f"mean_sigma_mag_comp_{region_name}"] = self._assemble_compiled(f"sigma_mag_comp_{region_name}") / vol
            else:
                data[f"mean_S_ff_{region_name}"] = 0.0
                data[f"mean_E_ff_{region_name}"] = 0.0
                if self.enable_decomp:
                    data[f"mean_S_ss_{region_name}"] = 0.0
                    data[f"mean_S_nn_{region_name}"] = 0.0
                    data[f"mean_E_ss_{region_name}"] = 0.0
                    data[f"mean_E_nn_{region_name}"] = 0.0

        return data

    def _calculate_incremental_work(self):
        """
        Calculates Work Densities (S : dE) using Trapezoidal Rule (0.5 * (S_new + S_old) : dE).

        Gated by enable_regional_internal. When disabled, returns an empty
        dict because compute_per_cell.py is the canonical source for these
        quantities.
        """
        if not self.enable_regional_internal:
            return {}

        # S_passive and S_comp are already interpolated by _calculate_state_variables()
        # which is always called before this method.

        # Strain increment (Interpolated is fine here, as Strain is continuous)
        dE = self.E_cur - self.E_prev
        
        # Trapezoidal: 0.5 * (S_curr + S_prev) : dE
        wd_total = 0.5 * ufl.inner(self.S_tot_ufl + self.S_prev, dE)
        wd_active = 0.5 * ufl.inner(self.S_act_ufl + self.S_active_prev, dE)
        wd_passive = 0.5 * ufl.inner(self.S_pas_ufl + self.S_passive_prev, dE)
        wd_comp = 0.5 * ufl.inner(self.S_cmp_ufl + self.S_comp_prev, dE)
        
        f0 = self.fiber_fields['f0']
        s0 = self.fiber_fields['s0'] 
        n0 = self.fiber_fields['n0'] 
        
        def proj(T, v): return ufl.inner(ufl.dot(T, v), v)
        
        # Directional Work (Trapezoidal)
        wd_fiber = 0.5 * (proj(self.S_tot_ufl, f0) + proj(self.S_prev, f0)) * proj(dE, f0)
        wd_sheet = 0.5 * (proj(self.S_tot_ufl, s0) + proj(self.S_prev, s0)) * proj(dE, s0)
        wd_normal = 0.5 * (proj(self.S_tot_ufl, n0) + proj(self.S_prev, n0)) * proj(dE, n0)
        
        # Shear is whatever remains
        wd_shear = wd_total - (wd_fiber + wd_sheet + wd_normal)
        wd_passive_fiber = 0.5 * (proj(self.S_pas_ufl, f0) + proj(self.S_passive_prev, f0)) * proj(dE, f0)

        # 3. Integration (using pre-compiled forms)
        data = {}
        regions = self._get_regions_to_integrate()

        for region_name, cell_tags, region_markers in regions:
            data[f"work_true_{region_name}"] = self._assemble_compiled(f"wd_total_{region_name}")
            data[f"work_active_{region_name}"] = self._assemble_compiled(f"wd_active_{region_name}")
            data[f"work_passive_{region_name}"] = self._assemble_compiled(f"wd_passive_{region_name}")
            data[f"work_comp_{region_name}"] = self._assemble_compiled(f"wd_comp_{region_name}")

            data[f"work_ff_{region_name}"] = self._assemble_compiled(f"wd_fiber_{region_name}")
            data[f"work_ss_{region_name}"] = self._assemble_compiled(f"wd_sheet_{region_name}")
            data[f"work_nn_{region_name}"] = self._assemble_compiled(f"wd_normal_{region_name}")
            data[f"work_cross_{region_name}"] = self._assemble_compiled(f"wd_shear_{region_name}")
            data[f"work_passive_ff_{region_name}"] = self._assemble_compiled(f"wd_passive_fiber_{region_name}")

        return data
    
    def _calculate_pressure_proxies(self, model_history, current_state=None):
        """
        Pure 0D PV loop work proxy: P_0D * dV_0D (clinical analogue).

        Uses 0D Windkessel pressure and unscaled 0D volume — the quantities
        a clinician would measure from echo + catheter.  No mixing of solver
        Lagrange-multiplier pressure with 0D volumes.
        """
        proxies = {}
        current_state = current_state or {}

        MMHG_TO_PA = 133.322
        ML_TO_M3 = 1e-6

        # Pure 0D quantities (fallback to solver values if 0D keys missing)
        p_LV_Pa = (current_state.get("p_LV_0D", 0.0) or 0.0) * MMHG_TO_PA
        p_RV_Pa = (current_state.get("p_RV_0D", 0.0) or 0.0) * MMHG_TO_PA
        V_LV_m3 = (current_state.get("V_LV_Clinical", 0.0) or 0.0) * ML_TO_M3
        V_RV_m3 = (current_state.get("V_RV_Clinical", 0.0) or 0.0) * ML_TO_M3

        # Initialization (first call — store previous, return zero work)
        if self.V_LV_prev is None:
            self.V_LV_prev = V_LV_m3
            self.V_RV_prev = V_RV_m3
            self.p_LV_0D_prev = p_LV_Pa
            self.p_RV_0D_prev = p_RV_Pa
            return {"work_proxy_pv_LV": 0.0, "work_proxy_pv_RV": 0.0}

        # Volume increments
        dV_LV = V_LV_m3 - self.V_LV_prev
        dV_RV = V_RV_m3 - self.V_RV_prev

        # Artifact protection: ignore dV > 10 mL in one step
        if abs(dV_LV) > 1e-5: dV_LV = 0.0
        if abs(dV_RV) > 1e-5: dV_RV = 0.0

        # Trapezoidal rule: W = P_avg * dV
        p_LV_avg = 0.5 * (p_LV_Pa + self.p_LV_0D_prev)
        p_RV_avg = 0.5 * (p_RV_Pa + self.p_RV_0D_prev)

        proxies["work_proxy_pv_LV"] = p_LV_avg * dV_LV
        proxies["work_proxy_pv_RV"] = p_RV_avg * dV_RV

        # Update previous state
        self.V_LV_prev = V_LV_m3
        self.V_RV_prev = V_RV_m3
        self.p_LV_0D_prev = p_LV_Pa
        self.p_RV_0D_prev = p_RV_Pa

        return proxies

    def _calculate_pressure_strain_work(self, current_state, mechanics_data):
        """
        Calculates Pressure-Strain Work Index (PSWI) with detailed Septal breakdown.
        Handles both original (LV/RV/Septum) and AHA sub-regions generically.
        """
        data = {}
        # Convert to Pa
        p_LV = (current_state.get("p_LV", 0.0) or 0.0) * 133.322
        p_RV = (current_state.get("p_RV", 0.0) or 0.0) * 133.322

        # Trapezoidal Pressures (Pa)
        p_LV_avg = 0.5 * (p_LV + self.p_LV_prev)
        p_RV_avg = 0.5 * (p_RV + self.p_RV_prev)

        # Region classification: which pressure applies to which region
        lv_regions = ["LV"]
        rv_regions = ["RV"]
        septum_regions = ["Septum"]

        if self.aha_tags is not None:
            lv_regions += ["Basal_LV", "Mid_LV"]
            rv_regions += ["Basal_RV", "Mid_RV"]
            septum_regions += ["Basal_Septum", "Mid_Septum"]

        for strain_key, suffix in [("E_ff", "ff"), ("E_ll", "ll")]:
            # LV-type: P_LV × dE
            for region in lv_regions:
                eps = mechanics_data.get(f"mean_{strain_key}_{region}", 0.0)
                eps_prev = self._eps_prev.get(f"{strain_key}_{region}", 0.0)
                dE = eps - eps_prev
                vol = self.region_volumes.get(region, 1.0)
                data[f"work_ps_{suffix}_{region}"] = (p_LV_avg * dE) * vol
                self._eps_prev[f"{strain_key}_{region}"] = eps

            # RV-type: P_RV × dE
            for region in rv_regions:
                eps = mechanics_data.get(f"mean_{strain_key}_{region}", 0.0)
                eps_prev = self._eps_prev.get(f"{strain_key}_{region}", 0.0)
                dE = eps - eps_prev
                vol = self.region_volumes.get(region, 1.0)
                data[f"work_ps_{suffix}_{region}"] = (p_RV_avg * dE) * vol
                self._eps_prev[f"{strain_key}_{region}"] = eps

            # Septum-type: Trans/PLV/PRV variants
            for region in septum_regions:
                eps = mechanics_data.get(f"mean_{strain_key}_{region}", 0.0)
                eps_prev = self._eps_prev.get(f"{strain_key}_{region}", 0.0)
                dE = eps - eps_prev
                vol = self.region_volumes.get(region, 1.0)
                data[f"work_ps_{suffix}_{region}_Trans"] = ((p_LV_avg - p_RV_avg) * dE) * vol
                data[f"work_ps_{suffix}_{region}_PLV"] = (p_LV_avg * dE) * vol
                data[f"work_ps_{suffix}_{region}_PRV"] = (p_RV_avg * dE) * vol
                data[f"work_ps_{suffix}_{region}_Mean"] = (0.5 * (p_LV_avg + p_RV_avg) * dE) * vol
                data[f"work_ps_{suffix}_{region}_Sum"] = ((p_LV_avg + p_RV_avg) * dE) * vol
                self._eps_prev[f"{strain_key}_{region}"] = eps

        return data
    
    def _calculate_robin_work(self):
        """
        Work done BY boundary springs ON the mesh.
        Uses pre-compiled forms from _precompile_forms().
        """
        wd_epi = self._assemble_compiled("robin_epi")
        wd_base = self._assemble_compiled("robin_base")
        return {"work_robin_epi": wd_epi, "work_robin_base": wd_base}

    def _resolve_lv_rv_tags(self, geo):
        """Return (lv_facet_tag, rv_facet_tag, ffun) or (None, None, None)."""
        markers = getattr(geo, "markers", None) or {}
        lv_names = ("ENDO_LV", "LV")
        rv_names = ("ENDO_RV", "RV")
        lv_tag = next((markers[n][0] for n in lv_names if n in markers), None)
        rv_tag = next((markers[n][0] for n in rv_names if n in markers), None)
        ffun = getattr(geo, "ffun", None)
        if lv_tag is None or rv_tag is None or ffun is None:
            return None, None, None
        return int(lv_tag), int(rv_tag), ffun

    def _local_centroids(self):
        """Return (n_local, 3) array of local cell centroids."""
        n_local = self.mesh.topology.index_map(3).size_local
        self.mesh.topology.create_connectivity(3, 0)
        c2v = self.mesh.topology.connectivity(3, 0)
        centroids = np.zeros((n_local, 3), dtype=np.float64)
        for c in range(n_local):
            vlist = c2v.links(c)
            centroids[c] = self.mesh.geometry.x[vlist].mean(axis=0)
        return centroids, n_local

    def _cells_to_meshtags(self, values, n_local):
        """Wrap a per-local-cell int32 array into a MeshTags on the mesh."""
        indices = np.arange(n_local, dtype=np.int32)
        return dolfinx.mesh.meshtags(self.mesh, 3, indices,
                                      values.astype(np.int32))

    def _build_tau_tags_euclid(self, geo):
        """(a) Euclidean tau = d_LV / (d_LV + d_RV), LV iff tau < 0.5."""
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            if self.rank == 0:
                print("MetricsCalculator: scipy not available; tau_tags_eu disabled")
            return None

        lv_tag, rv_tag, ffun = self._resolve_lv_rv_tags(geo)
        if ffun is None:
            if self.rank == 0:
                print("MetricsCalculator: tau_tags_eu disabled — LV/RV endo "
                      "markers missing")
            return None

        # Gather LV / RV endo vertex coordinates (global across ranks)
        self.mesh.topology.create_connectivity(2, 0)
        f2v = self.mesh.topology.connectivity(2, 0)

        def surface_vertices(tag):
            facets = ffun.find(int(tag))
            coords = [self.mesh.geometry.x[v]
                      for f in facets for v in f2v.links(f)]
            return (np.asarray(coords, dtype=np.float64)
                    if coords else np.empty((0, 3), dtype=np.float64))

        lv_local = surface_vertices(lv_tag)
        rv_local = surface_vertices(rv_tag)
        all_lv = self.comm.allgather(lv_local)
        all_rv = self.comm.allgather(rv_local)
        lv_global = (np.concatenate([a for a in all_lv if len(a)])
                     if any(len(a) for a in all_lv) else np.empty((0, 3)))
        rv_global = (np.concatenate([a for a in all_rv if len(a)])
                     if any(len(a) for a in all_rv) else np.empty((0, 3)))
        if len(lv_global) == 0 or len(rv_global) == 0:
            if self.rank == 0:
                print("MetricsCalculator: tau_tags_eu disabled — empty LV or RV surface")
            return None

        centroids, n_local = self._local_centroids()
        d_lv = cKDTree(lv_global).query(centroids)[0]
        d_rv = cKDTree(rv_global).query(centroids)[0]
        tau = d_lv / (d_lv + d_rv + 1e-18)
        values = np.where(tau < 0.5, 11, 12).astype(np.int32)
        tau_mt = self._cells_to_meshtags(values, n_local)

        n_lv_g = self.comm.allreduce(int((values == 11).sum()), op=MPI.SUM)
        n_rv_g = self.comm.allreduce(int((values == 12).sum()), op=MPI.SUM)
        if self.rank == 0:
            print(f"MetricsCalculator: tau_tags_eu (Euclidean d_LV/(d_LV+d_RV)<0.5) — "
                  f"LV_tau_eu={n_lv_g} cells, RV_tau_eu={n_rv_g} cells")
        return tau_mt

    def _build_tau_tags_laplace(self, geo):
        """(b) Laplace tau: solve ∇²u = 0 with u=0 on LV endo, u=1 on RV endo.
        LV iff tau_lap < 0.5."""
        try:
            from petsc4py import PETSc
        except ImportError:
            if self.rank == 0:
                print("MetricsCalculator: petsc4py not available; tau_tags_lap disabled")
            return None

        lv_tag, rv_tag, ffun = self._resolve_lv_rv_tags(geo)
        if ffun is None:
            if self.rank == 0:
                print("MetricsCalculator: tau_tags_lap disabled — LV/RV endo markers missing")
            return None

        V_CG1 = dolfinx.fem.functionspace(self.mesh, ("CG", 1))
        u_tri = ufl.TrialFunction(V_CG1)
        v_tst = ufl.TestFunction(V_CG1)
        a_lap = ufl.dot(ufl.grad(u_tri), ufl.grad(v_tst)) * ufl.dx
        L_zero = dolfinx.fem.Constant(self.mesh, 0.0) * v_tst * ufl.dx

        self.mesh.topology.create_connectivity(2, 0)
        lv_facets = ffun.find(lv_tag)
        rv_facets = ffun.find(rv_tag)
        lv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, lv_facets)
        rv_dofs = dolfinx.fem.locate_dofs_topological(V_CG1, 2, rv_facets)

        # LV endo → 0, RV endo → 1 (so LV_tau_lap: tau_lap < 0.5, matching
        # the Euclidean convention that LV cells are on the "small tau" side).
        try:
            problem = dolfinx.fem.petsc.LinearProblem(
                a_lap, L_zero,
                bcs=[
                    dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), lv_dofs, V_CG1),
                    dolfinx.fem.dirichletbc(PETSc.ScalarType(1.0), rv_dofs, V_CG1),
                ],
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                petsc_options_prefix="metrics_tau_lap",
            )
            tau_cg1 = problem.solve()
        except Exception as e:
            if self.rank == 0:
                print(f"MetricsCalculator: tau_tags_lap disabled — Laplace "
                      f"solve failed: {e}")
            return None

        # Interpolate to DG0 (one value per cell)
        V_DG0 = dolfinx.fem.functionspace(self.mesh, ("DG", 0))
        tau_dg0 = dolfinx.fem.Function(V_DG0)
        tau_dg0.interpolate(tau_cg1)
        n_local = self.mesh.topology.index_map(3).size_local
        tau_vals = tau_dg0.x.array[:n_local].copy()
        values = np.where(tau_vals < 0.5, 11, 12).astype(np.int32)
        tau_mt = self._cells_to_meshtags(values, n_local)

        n_lv_g = self.comm.allreduce(int((values == 11).sum()), op=MPI.SUM)
        n_rv_g = self.comm.allreduce(int((values == 12).sum()), op=MPI.SUM)
        if self.rank == 0:
            print(f"MetricsCalculator: tau_tags_lap (Laplace u<0.5, u=0 LV u=1 RV) — "
                  f"LV_tau_lap={n_lv_g} cells, RV_tau_lap={n_rv_g} cells")
        return tau_mt

    def _get_regions_to_integrate(self):
        regions = []
        if self.region_tags is not None:
            regions.append(("LV", self.region_tags, [1]))
            regions.append(("RV", self.region_tags, [2]))
            regions.append(("Septum", self.region_tags, [3]))
            regions.append(("Whole", self.region_tags, [1, 2, 3])) #no 4
        if self.tau_tags_eu is not None:
            regions.append(("LV_tau_eu", self.tau_tags_eu, [11]))
            regions.append(("RV_tau_eu", self.tau_tags_eu, [12]))
        if self.tau_tags_lap is not None:
            regions.append(("LV_tau_lap", self.tau_tags_lap, [11]))
            regions.append(("RV_tau_lap", self.tau_tags_lap, [12]))
        if self.aha_tags is not None:
            regions.append(("Basal_LV", self.aha_tags, [1]))
            regions.append(("Basal_RV", self.aha_tags, [2]))
            regions.append(("Basal_Septum", self.aha_tags, [3]))
            regions.append(("Mid_LV", self.aha_tags, [4]))
            regions.append(("Mid_RV", self.aha_tags, [5]))
            regions.append(("Mid_Septum", self.aha_tags, [6]))
        return regions

    def setup_csv_logging(self, file_path):
        if self.rank != 0: return
        self.trace_path = Path(file_path)
        self.trace_path.unlink(missing_ok=True)   # reset any previous trace
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
            # Store region volumes as scalar metadata (m^3)
            for rname, rvol in self.region_volumes.items():
                downsampled[f"region_volume_{rname}"] = float(rvol)
            np.save(output_dir / f"metrics_downsample_{factor}.npy", downsampled, allow_pickle=True)

    def compute_regional_metrics(self, timestep_idx, t, model_history, skip_work_calc=False, current_state=None):
        metrics = {}

        # Initialize pressure history on first call (independent of has_previous_state)
        # This prevents the trapezoidal average from using p_prev=0 at the first work step.
        if not self._p_initialized and current_state:
            self.p_LV_prev = (current_state.get("p_LV", 0.0) or 0.0) * 133.322
            self.p_RV_prev = (current_state.get("p_RV", 0.0) or 0.0) * 133.322
            self._p_initialized = True

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
            
            # E. Boundary Work via Surface Integration (Exact)
            metrics.update(self._calculate_boundary_work_exact(current_state))
            
            if self.rank == 0:
                w_tot = metrics.get("work_true_LV", 0.0)
                ps_idx = metrics.get("work_ps_ff_LV", 0.0)
                print(f"STATS | t={t:.3f} | W_Tot={w_tot:.4e} | PS_Idx={ps_idx:.4e}")

        elif not self.has_previous_state:
            # Pressure-strain work is a loop over strain increments. Clinical
            # strain is zeroed at the start of the analysed beat, so the first
            # sampled FEM strain state must become the previous state for the
            # next increment rather than allowing the first work step to jump
            # from an artificial zero-strain unloaded reference.
            for strain_key in ("E_ff", "E_ll"):
                prefix = f"mean_{strain_key}_"
                for key, value in state_data.items():
                    if key.startswith(prefix):
                        self._eps_prev[f"{strain_key}_{key[len(prefix):]}"] = value
            self.eps_LV_prev = self._eps_prev.get("E_ff_LV", 0.0)
            self.eps_RV_prev = self._eps_prev.get("E_ff_RV", 0.0)
            self.eps_Septum_prev = self._eps_prev.get("E_ff_Septum", 0.0)
            self.eps_ll_LV_prev = self._eps_prev.get("E_ll_LV", 0.0)
            self.eps_ll_RV_prev = self._eps_prev.get("E_ll_RV", 0.0)
            self.eps_ll_Septum_prev = self._eps_prev.get("E_ll_Septum", 0.0)

            # Initialize all work keys to 0.0 at Step 0 to "reserve" the CSV columns.
            # We perform a dummy call to get the keys, but we don't use the values.
            # Note: This requires temporary state setup or manual key definition.
            # Safer approach: Manually define the expected keys based on regions.
            region_suffixes = [r[0] for r in self._get_regions_to_integrate()]

            # Regional internal work keys — only reserved when the gate is on.
            # When disabled, downstream consumers should read per-cell work
            # from compute_per_cell.py's NPZ instead.
            if self.enable_regional_internal:
                prefixes = ["work_true", "work_active", "work_passive", "work_comp",
                            "work_ff", "work_ss", "work_nn", "work_cross", "work_passive_ff"]
                for r in region_suffixes:
                    for p in prefixes:
                        metrics[f"{p}_{r}"] = 0.0

            # Robin keys
            metrics["work_robin_epi"] = 0.0
            metrics["work_robin_base"] = 0.0

            # PV/PS keys — generate dynamically to cover AHA regions too
            metrics["work_proxy_pv_LV"] = 0.0
            metrics["work_proxy_pv_RV"] = 0.0

            lv_regions = ["LV"]
            rv_regions = ["RV"]
            septum_regions = ["Septum"]
            if self.aha_tags is not None:
                lv_regions += ["Basal_LV", "Mid_LV"]
                rv_regions += ["Basal_RV", "Mid_RV"]
                septum_regions += ["Basal_Septum", "Mid_Septum"]

            for suffix in ["ff", "ll"]:
                for region in lv_regions + rv_regions:
                    metrics[f"work_ps_{suffix}_{region}"] = 0.0
                for region in septum_regions:
                    metrics[f"work_ps_{suffix}_{region}_Trans"] = 0.0
                    metrics[f"work_ps_{suffix}_{region}_PLV"] = 0.0
                    metrics[f"work_ps_{suffix}_{region}_PRV"] = 0.0
                    metrics[f"work_ps_{suffix}_{region}_Mean"] = 0.0
                    metrics[f"work_ps_{suffix}_{region}_Sum"] = 0.0
            
            #exact work keys
            metrics["work_boundary_exact_LV"] = 0.0
            metrics["work_boundary_exact_RV"] = 0.0
            # Compute initial FEM cavity volumes (m³ -> mL)
            V_LV_init = self._assemble_compiled("vol_cur_LV") if "vol_cur_LV" in self._compiled_forms else 0.0
            V_RV_init = self._assemble_compiled("vol_cur_RV") if "vol_cur_RV" in self._compiled_forms else 0.0
            metrics["V_LV_FEM"] = V_LV_init * 1e6
            metrics["V_RV_FEM"] = V_RV_init * 1e6
        
        #print(metrics)
        
        return metrics
    
    def _calculate_boundary_work_exact(self, current_state):
        """
        Exact external work: W = -P_avg * DeltaV_cavity
        Uses pre-compiled volume forms from _precompile_forms().
        """
        # Pressure (Convert mmHg -> Pa)
        p_LV = (current_state.get("p_LV", 0.0) or 0.0) * 133.322
        p_RV = (current_state.get("p_RV", 0.0) or 0.0) * 133.322

        # Trapezoidal Average Pressure
        p_LV_avg = 0.5 * (p_LV + self.p_LV_prev)
        p_RV_avg = 0.5 * (p_RV + self.p_RV_prev)

        # Update History
        self.p_LV_prev = p_LV
        self.p_RV_prev = p_RV

        # Compute V_new and V_old for each cavity using pre-compiled forms
        V_LV_cur  = self._assemble_compiled("vol_cur_LV")
        V_LV_prev = self._assemble_compiled("vol_prev_LV")
        V_RV_cur  = self._assemble_compiled("vol_cur_RV")
        V_RV_prev = self._assemble_compiled("vol_prev_RV")

        w_bnd_LV = p_LV_avg * (V_LV_cur - V_LV_prev)
        w_bnd_RV = p_RV_avg * (V_RV_cur - V_RV_prev)

        # Convert cavity volumes from m³ to mL for storage
        volume_m3_to_ml = 1e6

        return {
            "work_boundary_exact_LV": w_bnd_LV,
            "work_boundary_exact_RV": w_bnd_RV,
            "V_LV_FEM": V_LV_cur * volume_m3_to_ml,
            "V_RV_FEM": V_RV_cur * volume_m3_to_ml,
        }
