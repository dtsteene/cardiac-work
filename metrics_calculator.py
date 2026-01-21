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
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from mpi4py import MPI
import dolfinx
import ufl


class MetricsCalculator:
    """Calculate cardiac mechanics metrics per region at each timestep."""

    def __init__(self, geometry, geo, fiber_field_map, problem, comm, cardiac_model):
        """
        Args:
            geometry: pulse.HeartGeometry object
            geo: cardiac_geometries.Geometry object (for access to mesh tags)
            fiber_field_map: dict with fiber vectors in current configuration
                {'f0': f0_mapped, 's0': s0_mapped, 'l0': long, 'c0': circ, 'n0': norm}
            problem: pulse.problem.StaticProblem with problem.u, cavities, etc.
            comm: MPI communicator
            cardiac_model: pulse.CardiacModel - THE single source of truth for all stress
                Contains material, active, and compressibility components
        """
        self.geometry = geometry
        self.geo = geo
        self.fiber_fields = fiber_field_map
        self.cardiac_model = cardiac_model  # GRAND UNIFICATION: Single model
        self.problem = problem
        self.comm = comm
        self.rank = comm.rank

        self.mesh = geometry.mesh
        self.volume2ml = 1e6

        # Pre-compile UFL expressions for efficiency
        self._setup_expressions()

        # Storage for metrics at each timestep
        # Use regular dict with list values instead of nested defaultdicts to avoid append issues
        self.metrics_history = defaultdict(list)

        # Track if we have previous state
        self.has_previous_state = False
        self.S_prev = None
        self.E_prev = None
        self.u_prev = None

        # Track previous volumes for proxy work calculation
        self.V_LV_prev = None
        self.V_RV_prev = None

        # Access cell tags (septum: LV/RV/Septum) and AHA tags if available
        self.septum_tags = geo.additional_data.get("markers_mt", None)
        self.aha_tags = None  # May be loaded later if available

    def _setup_expressions(self):
        """Pre-compile UFL expressions for stress/strain computation."""
        f0 = self.fiber_fields['f0']
        s0 = self.fiber_fields['s0']

        # Deformation gradient and associated quantities
        u = self.problem.u
        I = ufl.Identity(3)
        F = ufl.variable(ufl.grad(u) + I)
        C = F.T * F
        E = 0.5 * (C - I)

        # NOTE: We do NOT precompute full S here because it depends on Ta which varies.
        # We compute S fresh in _calculate_true_work() during each timestep.
        # For visualization, use passive material component only.
        
        S_passive = self.cardiac_model.material.S(ufl.variable(C))
        sigma_passive = self.cardiac_model.material.sigma(F)

        # Mapped fiber direction in current config
        f_current = (F * f0) / ufl.sqrt(ufl.inner(F * f0, F * f0))

        # Scalar projections (using passive for visualization)
        fiber_stress = ufl.inner(sigma_passive * f_current, f_current)
        fiber_strain = ufl.inner(E * f0, f0)

        # Store for later use
        self.u = u
        self.F = F
        self.C = C  # Store C for computing S later
        self.E = E
        self.fiber_stress_expr = fiber_stress
        self.fiber_strain_expr = fiber_strain

    def load_aha_tags(self, checkpoint_file):
        """Load AHA tags from checkpoint if available."""
        try:
            import adios2
            with adios2.FileReader(str(checkpoint_file)) as fr:
                # Try to read AHA meshtags if they exist
                # This is a fallback; usually stored separately
                pass
        except:
            if self.rank == 0:
                print("Warning: Could not load AHA tags. Proceeding with septum tags only.")

    def update_state(self):
        """Update previous state for work calculation. Call AFTER each solve."""
        W_tensor = dolfinx.fem.functionspace(self.mesh, ("DG", 1, (3, 3)))

        # Create/update S and E in DG1 space
        if self.S_prev is None:
            self.S_prev = dolfinx.fem.Function(W_tensor)
            self.E_prev = dolfinx.fem.Function(W_tensor)
            self.u_prev = dolfinx.fem.Function(self.problem.u.function_space)

        # Compute current FULL stress (must be fresh each time due to Ta)
        S_full = self.cardiac_model.S(ufl.variable(self.C))
        
        # Interpolate current S and E
        S_expr = dolfinx.fem.Expression(S_full, W_tensor.element.interpolation_points)
        E_expr = dolfinx.fem.Expression(self.E, W_tensor.element.interpolation_points)

        S_cur = dolfinx.fem.Function(W_tensor)
        E_cur = dolfinx.fem.Function(W_tensor)
        S_cur.interpolate(S_expr)
        E_cur.interpolate(E_expr)

        # Store current as previous for next iteration
        self.S_prev.x.array[:] = S_cur.x.array
        self.E_prev.x.array[:] = E_cur.x.array
        self.u_prev.x.array[:] = self.problem.u.x.array

        self.has_previous_state = True

    def compute_regional_metrics(self, timestep_idx, t, model_history,
                                 skip_work_calc=False, current_state=None):
        """
        Compute all metrics for current timestep and all regions.

        Args:
            timestep_idx: Current index in simulation
            t: Current time
            model_history: Circulation model history dict with p_LV, p_RV, V_LV, V_RV, etc.
            skip_work_calc: If True (first step), skip work calculations
            current_state: Optional dict {'p_LV': val, 'V_LV': val, ...} to override history

        Returns:
            metrics_dict: {region_name: {metric_name: value}}
        """
        metrics = {}

        if not skip_work_calc and self.has_previous_state:
            # === TRUE WORK (Stress-based, Internal) ===
            metrics.update(self._calculate_true_work())

            # === ACTIVE vs PASSIVE SPLIT ===
            metrics.update(self._calculate_active_passive_work())

            # === BOUNDARY WORK (External, for validation) ===
            metrics.update(self._calculate_boundary_work(current_state))

            # === PRESSURE PROXIES (PV-based) ===
            metrics.update(self._calculate_pressure_proxies(model_history, current_state))

            # === PRESSURE-STRAIN AREA (PSA - new clinical proxy) ===
            metrics.update(self._calculate_pressure_strain_area(current_state))

            # --- SUMMARY LOG ---
            if self.comm.rank == 0:
                # Flat summary per timestep to spot anomalies quickly
                p_LV = current_state.get("p_LV", model_history.get("p_LV", [0.0])[-1] if "p_LV" in model_history else 0.0)
                p_RV = current_state.get("p_RV", model_history.get("p_RV", [0.0])[-1] if "p_RV" in model_history else 0.0)
                V_LV = current_state.get("V_LV", model_history.get("V_LV", [0.0])[-1] if "V_LV" in model_history else 0.0)
                V_RV = current_state.get("V_RV", model_history.get("V_RV", [0.0])[-1] if "V_RV" in model_history else 0.0)
                w_true_lv = metrics.get("work_true_LV", 0.0)
                w_true_rv = metrics.get("work_true_RV", 0.0)
                w_active_lv = metrics.get("work_active_LV", 0.0)
                w_passive_lv = metrics.get("work_passive_LV", 0.0)
                w_boundary_lv = metrics.get("work_boundary_LV", 0.0)
                w_proxy_lv = metrics.get("work_proxy_pv_LV", 0.0)
                psa_lv = metrics.get("psa_LV", 0.0)
                print(
                    "METRICS STEP | "
                    f"i={timestep_idx}, t={t:.4f} s | "
                    f"p_LV={p_LV:.2f} mmHg, V_LV={V_LV:.2f} mL | "
                    f"True={w_true_lv:.3e}, Active={w_active_lv:.3e}, Passive={w_passive_lv:.3e}, "
                    f"Boundary={w_boundary_lv:.3e}, Proxy={w_proxy_lv:.3e}, PSA={psa_lv:.3e} J"
                )

        return metrics

    def _calculate_true_work(self):
        """Calculate true work using proper UFL integration."""
        W_tensor = dolfinx.fem.functionspace(self.mesh, ("DG", 1, (3, 3)))

        # Compute FULL stress (Passive + Active + Pressure) at current timestep
        # This must be done fresh each time because Ta changes
        S_full = self.cardiac_model.S(ufl.variable(self.C))
        
        # Current S and E
        S_expr = dolfinx.fem.Expression(S_full, W_tensor.element.interpolation_points)
        E_expr = dolfinx.fem.Expression(self.E, W_tensor.element.interpolation_points)

        S_cur = dolfinx.fem.Function(W_tensor)
        E_cur = dolfinx.fem.Function(W_tensor)
        S_cur.interpolate(S_expr)
        E_cur.interpolate(E_expr)

        # Work component form: 0.5 * (S_prev + S_cur) : (E_cur - E_prev)
        dS_avg = 0.5 * (self.S_prev + S_cur)
        dE = E_cur - self.E_prev
        W_density = ufl.inner(dS_avg, dE)

        # === DEBUG: Calculate Passive-Only Work for comparison ===
        # Get passive stress from material component only (strips out active + pressure)
        S_passive_only = self.cardiac_model.material.S(ufl.variable(self.C))
        S_passive_only_expr = dolfinx.fem.Expression(S_passive_only, W_tensor.element.interpolation_points)
        S_passive_only_cur = dolfinx.fem.Function(W_tensor)
        S_passive_only_cur.interpolate(S_passive_only_expr)
        
        # Store previous passive stress for incremental calculation
        if not hasattr(self, 'S_passive_prev'):
            self.S_passive_prev = dolfinx.fem.Function(W_tensor)
            self.S_passive_prev.x.array[:] = S_passive_only_cur.x.array
        
        dS_passive_avg = 0.5 * (self.S_passive_prev + S_passive_only_cur)
        W_passive_density = ufl.inner(dS_passive_avg, dE)

        work_data = {}
        regions_to_integrate = self._get_regions_to_integrate()

        metadata = {"quadrature_degree": 4}

        for region_name, cell_tags, region_markers in regions_to_integrate:
            if cell_tags is None:
                continue

            # Define Measure for this specific tag set
            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata=metadata)

            total_work_local = 0.0
            passive_work_local = 0.0

            for marker_val in region_markers:
                try:
                    # Integrate TOTAL work (Full stress: Passive + Active + Pressure)
                    form_work = dolfinx.fem.form(W_density * dx_sub(int(marker_val)))
                    val_total = dolfinx.fem.assemble_scalar(form_work)
                    total_work_local += val_total
                    
                    # Integrate PASSIVE-ONLY work (Material only)
                    form_passive = dolfinx.fem.form(W_passive_density * dx_sub(int(marker_val)))
                    val_passive = dolfinx.fem.assemble_scalar(form_passive)
                    passive_work_local += val_passive
                    
                except Exception as e:
                    if self.rank == 0:
                        print(f"Error integrating work for {region_name}, marker {marker_val}: {e}")

            # Global Sum
            total_work_global = self.comm.allreduce(total_work_local, op=MPI.SUM)
            passive_work_global = self.comm.allreduce(passive_work_local, op=MPI.SUM)
            
            # === DEBUG LOGGING: Show Active Contribution ===
            if self.rank == 0 and region_name in ["LV", "RV"]:
                diff = total_work_global - passive_work_global
                ratio = abs(diff/passive_work_global) if abs(passive_work_global) > 1e-12 else 0.0
                print(
                    f"DEBUG WORK SPLIT ({region_name}) | "
                    f"Total={total_work_global:.3e} J, "
                    f"PassiveOnly={passive_work_global:.3e} J, "
                    f"Diff(Active+Pressure)={diff:.3e} J, "
                    f"Ratio={ratio:.2f}x"
                )

            # Store TOTAL ENERGY (Joules), do not divide by volume
            work_data[f"work_true_{region_name}"] = total_work_global

        # Update passive stress for next timestep
        self.S_passive_prev.x.array[:] = S_passive_only_cur.x.array

        return work_data

    def _calculate_pressure_proxies(self, model_history, current_state=None):
        """Calculate clinical pressure-based work proxies."""
        proxies = {}
        current_state = current_state or {}

        # Extract latest pressures (prefer override, then history)
        p_LV = current_state.get("p_LV", model_history.get("p_LV", [0.0])[-1] if "p_LV" in model_history else 0.0)
        p_RV = current_state.get("p_RV", model_history.get("p_RV", [0.0])[-1] if "p_RV" in model_history else 0.0)

        # Volumes: ALWAYS prefer explicit current_state (FEM cavity volumes) over history.
        # current_state is populated from lv_volume/rv_volume FEM constants, which are
        # the ground truth. History is secondary/optional.
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

        # Initialize on the very first call
        if self.V_LV_prev is None:
            self.V_LV_prev = V_LV
            self.V_RV_prev = V_RV
            proxies["work_proxy_pv_LV"] = 0.0
            proxies["work_proxy_pv_RV"] = 0.0
            proxies["work_proxy_pv_Septum"] = 0.0
            mmHg_mL_to_J = 1.33322e-4
            for key in list(proxies.keys()):
                proxies[key] *= mmHg_mL_to_J
            if self.comm.rank == 0:
                print(
                    "PROXY INIT | "
                    f"V_LV={V_LV:.3f}, V_RV={V_RV:.3f}, "
                    f"p_LV={p_LV:.2f} mmHg, p_RV={p_RV:.2f} mmHg"
                )
            return proxies

        # Calculate volume changes from PREVIOUS timestep (stored in instance vars)
        dV_LV = V_LV - self.V_LV_prev
        dV_RV = V_RV - self.V_RV_prev

        # Simple proxy: P * dV (pressure times volume change)
        # This is the standard clinical proxy for mechanical work
        proxies["work_proxy_pv_LV"] = p_LV * dV_LV if abs(dV_LV) > 1e-12 else 0.0
        proxies["work_proxy_pv_RV"] = p_RV * dV_RV if abs(dV_RV) > 1e-12 else 0.0

        if self.comm.rank == 0:
            print(
                "PROXY STEP | "
                f"V_LV_prev={self.V_LV_prev:.3f}, V_LV={V_LV:.3f}, dV_LV={dV_LV:.3e} mL, "
                f"p_LV={p_LV:.2f} mmHg, raw_LV={proxies['work_proxy_pv_LV']:.3e} mmHg*mL | "
                f"V_RV_prev={self.V_RV_prev:.3f}, V_RV={V_RV:.3f}, dV_RV={dV_RV:.3e} mL, "
                f"p_RV={p_RV:.2f} mmHg, raw_RV={proxies['work_proxy_pv_RV']:.3e} mmHg*mL"
            )

        # Septum and neutral regions: average pressure proxy
        p_avg = (p_LV + p_RV) / 2.0 if (p_LV + p_RV) > 0 else 0.0
        dV_total = dV_LV + dV_RV
        proxies["work_proxy_pv_Septum"] = p_avg * dV_total / 2.0 if abs(dV_total) > 1e-12 else 0.0

        # Convert from mmHg*mL to Joules for consistent comparison with stress-based work.
        # 1 mmHg = 133.322 Pa, 1 mL = 1e-6 m^3 → 1 mmHg*mL = 1.33322e-4 J.
        mmHg_mL_to_J = 1.33322e-4
        for key in list(proxies.keys()):
            proxies[key] *= mmHg_mL_to_J

# Update previous volumes for next iteration (store in instance vars)
        self.V_LV_prev = V_LV
        self.V_RV_prev = V_RV

        return proxies

    def _calculate_boundary_work(self, current_state=None):
        """
        Calculate external boundary work: W_ext = ∫ (p n · Δu) dA on endocardial surfaces.
        Looks up facet tags from geo.ffun / geo.markers with hardcoded fallback based on ParaView findings.
        """
        current_state = current_state or {}
        p_LV = current_state.get("p_LV", 0.0)
        p_RV = current_state.get("p_RV", 0.0)

        u_cur = self.problem.u
        if self.u_prev is None:
            Du = ufl.as_vector([0, 0, 0])
        else:
            Du = u_cur - self.u_prev

        boundary_work = {}

        try:
            # Priority 1: Pulse HeartGeometry standard location
            cavity_tags = getattr(self.geometry, "facet_tags", None)
            
            # Priority 2: Cardiac Geometries location
            if cavity_tags is None:
                cavity_tags = getattr(self.geo, "ffun", None)
            
            # Priority 3: Additional data
            if cavity_tags is None:
                cavity_tags = self.geo.additional_data.get("markers_facets", None)

            if cavity_tags is None:
                if self.rank == 0:
                    print("BOUNDARY WORK: Cavity facet tags (ffun) not available")
                return {"work_boundary_LV": 0.0, "work_boundary_RV": 0.0}

            ds_cav = ufl.Measure("ds", domain=self.mesh, subdomain_data=cavity_tags, metadata={"quadrature_degree": 4})
            n_vec = ufl.FacetNormal(self.mesh)

            # Dynamically resolve markers with HARDCODED FALLBACK based on ParaView verification
            try:
                lv_marker = self.geo.markers["ENDO_LV"][0]
            except (KeyError, TypeError, IndexError):
                lv_marker = 1  # Verified in ParaView as LV Endocardium
            
            try:
                rv_marker = self.geo.markers["ENDO_RV"][0]
            except (KeyError, TypeError, IndexError):
                rv_marker = 2  # Verified in ParaView as RV Endocardium

            if self.rank == 0:
                print(f"BOUNDARY WORK: Using markers LV={lv_marker}, RV={rv_marker}")

            # LV boundary work
            try:
                integrand_lv = p_LV * ufl.dot(n_vec, Du)
                form_lv = dolfinx.fem.form(integrand_lv * ds_cav(lv_marker))
                
                # DEBUG: Print boundary area to verify integration domain
                area_lv = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ds_cav(lv_marker)))
                if self.rank == 0:
                    print(f"DEBUG: LV Boundary Area (Tag {lv_marker}) = {area_lv:.6f} m^2")
                
                w_boundary_lv_local = dolfinx.fem.assemble_scalar(form_lv)
                w_boundary_lv_global = self.comm.allreduce(w_boundary_lv_local, op=MPI.SUM)
                boundary_work["work_boundary_LV"] = w_boundary_lv_global
                
                if self.rank == 0:
                    print(f"DEBUG: LV Boundary Work = {w_boundary_lv_global:.6e} J")
            except Exception as e:
                if self.rank == 0:
                    print(f"LV boundary work calculation error (Marker {lv_marker}): {e}")
                boundary_work["work_boundary_LV"] = 0.0

            # RV boundary work
            try:
                integrand_rv = p_RV * ufl.dot(n_vec, Du)
                form_rv = dolfinx.fem.form(integrand_rv * ds_cav(rv_marker))
                
                # DEBUG: Print boundary area to verify integration domain
                area_rv = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ds_cav(rv_marker)))
                if self.rank == 0:
                    print(f"DEBUG: RV Boundary Area (Tag {rv_marker}) = {area_rv:.6f} m^2")
                
                w_boundary_rv_local = dolfinx.fem.assemble_scalar(form_rv)
                w_boundary_rv_global = self.comm.allreduce(w_boundary_rv_local, op=MPI.SUM)
                boundary_work["work_boundary_RV"] = w_boundary_rv_global
                
                if self.rank == 0:
                    print(f"DEBUG: RV Boundary Work = {w_boundary_rv_global:.6e} J")
            except Exception as e:
                if self.rank == 0:
                    print(f"RV boundary work calculation error (Marker {rv_marker}): {e}")
                boundary_work["work_boundary_RV"] = 0.0

            if self.rank == 0:
                print(
                    f"BOUNDARY WORK | "
                    f"W_ext_LV={boundary_work.get('work_boundary_LV', 0.0):.3e} J, "
                    f"W_ext_RV={boundary_work.get('work_boundary_RV', 0.0):.3e} J"
                )

        except Exception as e:
            if self.rank == 0:
                print(f"Boundary work integration failed: {e}")
            boundary_work = {"work_boundary_LV": 0.0, "work_boundary_RV": 0.0}

        return boundary_work

    def _calculate_active_passive_work(self):
        """
        Split internal work into Active (contraction) and Passive (elastic) components.
        Active: stress from active contraction (Ta-related)
        Passive: stress from elastic recoil

        Simplified approach: assume activation scales active stress.
        More rigorous approach: decompose S = S_active + S_passive using material model.
        """
        W_tensor = dolfinx.fem.functionspace(self.mesh, ("DG", 1, (3, 3)))

        # Current S and E
        S_expr = dolfinx.fem.Expression(self.S, W_tensor.element.interpolation_points)
        E_expr = dolfinx.fem.Expression(self.E, W_tensor.element.interpolation_points)

        S_cur = dolfinx.fem.Function(W_tensor)
        E_cur = dolfinx.fem.Function(W_tensor)
        S_cur.interpolate(S_expr)
        E_cur.interpolate(E_expr)

        dE = E_cur - self.E_prev

        # For now: simple heuristic—magnitude of stress change relates to active/passive split
        # In future: use material model decomposition or activation function
        # W_active ≈ σ_active : dE, W_passive ≈ σ_passive : dE

        work_split = {}
        regions_to_integrate = self._get_regions_to_integrate()
        metadata = {"quadrature_degree": 4}

        for region_name, cell_tags, region_markers in regions_to_integrate:
            if cell_tags is None:
                continue

            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata=metadata)

            # Simplified: scale by stress magnitude (heuristic)
            # In a full model, separate S into active and passive components
            S_mag = ufl.sqrt(ufl.inner(self.S_prev, self.S_prev) + 1e-10)

            # Active fraction (0-1): simple sigmoid of stress magnitude
            S_ref = 10.0e3  # Reference stress in Pa
            active_frac = ufl.tanh(S_mag / S_ref)

            # Work densities
            W_avg = 0.5 * (self.S_prev + S_cur)
            W_total = ufl.inner(W_avg, dE)
            W_active_density = active_frac * W_total
            W_passive_density = (1.0 - active_frac) * W_total

            w_active_local = 0.0
            w_passive_local = 0.0

            for marker_val in region_markers:
                try:
                    form_active = dolfinx.fem.form(W_active_density * dx_sub(int(marker_val)))
                    form_passive = dolfinx.fem.form(W_passive_density * dx_sub(int(marker_val)))
                    w_active_local += dolfinx.fem.assemble_scalar(form_active)
                    w_passive_local += dolfinx.fem.assemble_scalar(form_passive)
                except Exception as e:
                    if self.rank == 0:
                        print(f"Error splitting work for {region_name}: {e}")

            w_active_global = self.comm.allreduce(w_active_local, op=MPI.SUM)
            w_passive_global = self.comm.allreduce(w_passive_local, op=MPI.SUM)

            work_split[f"work_active_{region_name}"] = w_active_global
            work_split[f"work_passive_{region_name}"] = w_passive_global

        if self.rank == 0:
            print(
                f"ACTIVE/PASSIVE | "
                f"LV_active={work_split.get('work_active_LV', 0.0):.3e} J, "
                f"LV_passive={work_split.get('work_passive_LV', 0.0):.3e} J"
            )

        return work_split

    def _calculate_pressure_strain_area(self, current_state=None):
        """
        Calculate Pressure-Strain Area (PSA): Cavity Pressure × Fiber Strain.
        Standard clinical proxy that may correlate better with regional mechanics.

        PSA = ∫ P · ε_ff dV per region (and AHA segment).
        """
        current_state = current_state or {}
        p_LV = current_state.get("p_LV", 0.0)
        p_RV = current_state.get("p_RV", 0.0)

        psa_metrics = {}
        regions_to_integrate = self._get_regions_to_integrate()
        metadata = {"quadrature_degree": 4}

        # Fiber strain (already computed in setup_expressions)
        f0 = self.fiber_fields['f0']
        fiber_strain_expr = ufl.inner(self.E * f0, f0)

        for region_name, cell_tags, region_markers in regions_to_integrate:
            if cell_tags is None:
                continue

            dx_sub = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags, metadata=metadata)

            # Select pressure based on region
            if "LV" in region_name or region_name in ["LV", "AHA_0", "AHA_1", "AHA_2", "AHA_3", "AHA_4"]:
                p_region = p_LV
            elif "RV" in region_name or region_name in ["AHA_5", "AHA_6"]:
                p_region = p_RV
            else:
                p_region = (p_LV + p_RV) / 2.0  # Septum: average

            psa_integrand = p_region * fiber_strain_expr

            psa_local = 0.0
            for marker_val in region_markers:
                try:
                    form_psa = dolfinx.fem.form(psa_integrand * dx_sub(int(marker_val)))
                    psa_local += dolfinx.fem.assemble_scalar(form_psa)
                except Exception as e:
                    if self.rank == 0:
                        print(f"Error computing PSA for {region_name}: {e}")

            psa_global = self.comm.allreduce(psa_local, op=MPI.SUM)
            psa_metrics[f"psa_{region_name}"] = psa_global

        if self.rank == 0:
            print(
                f"PSA | "
                f"PSA_LV={psa_metrics.get('psa_LV', 0.0):.3e} Pa, "
                f"PSA_RV={psa_metrics.get('psa_RV', 0.0):.3e} Pa"
            )

        return psa_metrics

    def _get_regions_to_integrate(self):
        """
        Return list of (region_name, cell_tags, markers) tuples for integration.

        Includes both septum tags (LV/RV/Septum) and AHA (if available).
        """
        regions = []

        # Septum tags
        if self.septum_tags is not None:
            regions.append(("LV", self.septum_tags, np.array([1])))
            regions.append(("RV", self.septum_tags, np.array([2])))
            regions.append(("Septum", self.septum_tags, np.array([3])))

        # AHA tags (0-6) if loaded
        if self.aha_tags is not None:
            for label in range(0, 7):
                regions.append((f"AHA_{label}", self.aha_tags, np.array([label])))

        return regions

    def _compute_region_volume(self, cell_tags, region_markers):
        """Compute volume of a region given cell tags and marker values."""
        if cell_tags is None:
            return 0.0

        # Create indicator function for the region
        V = dolfinx.fem.functionspace(self.mesh, ("DG", 0))
        indicator = dolfinx.fem.Function(V)

        # Set to 1 in region, 0 elsewhere
        for val in region_markers:
            cell_indices = np.where(cell_tags.values == val)[0]
            indicator.x.array[cell_indices] = 1.0

        # Integrate
        vol_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(indicator * ufl.dx)
        )
        vol_global = self.comm.allreduce(vol_local, op=MPI.SUM)

        return vol_global

    def store_metrics(self, region_metrics, timestep_idx, t, downsample_factor=1):
        """
        Store metrics to history. Can downsample if needed.

        Args:
            region_metrics: {metric_name: value} (flat dict from compute_regional_metrics)
            timestep_idx: Current timestep index
            t: Current time
            downsample_factor: Only store every Nth timestep (default: 1 = store all)
        """
        if timestep_idx % downsample_factor != 0:
            return

        self.metrics_history["time"].append(t)
        self.metrics_history["timestep"].append(timestep_idx)

        # region_metrics is already flat: {metric_name: value}
        for metric_name, value in region_metrics.items():
            self.metrics_history[metric_name].append(value)

    def save_metrics(self, output_dir, downsample_factors=None):
        """
        Save all metrics to file.

        Args:
            output_dir: Path to save
            downsample_factors: List of downsampling factors to save
                              (e.g., [1, 5, 10] saves full, every 5th, every 10th)
        """
        if self.rank != 0:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if downsample_factors is None:
            downsample_factors = [1]

        for factor in downsample_factors:
            # Downsample the data
            downsampled = {}
            indices = np.arange(0, len(self.metrics_history["time"]), factor)

            for key, values in self.metrics_history.items():
                if isinstance(values, list):
                    downsampled[key] = [values[i] for i in indices if i < len(values)]

            # Save
            filename = output_dir / f"metrics_downsample_{factor}.npy"
            np.save(filename, downsampled, allow_pickle=True)
            print(f"✓ Saved metrics (downsample={factor}) to {filename}")