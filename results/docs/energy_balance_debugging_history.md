# The Energy Balance Debugging Journey

**A chronological account of achieving correct stress-strain energy integration in a FEniCSx cardiac mechanics simulation.**

This document traces the evolution of the `MetricsCalculator` from January 20 to March 6, 2026 — a period during which the internal work integral `W_int = integral S:dE dV` was systematically brought into agreement with the external work `W_ext` applied by cavity pressures and boundary springs.

---

## 1. The Problem Statement

In finite element cardiac mechanics, the First Law of Thermodynamics requires:

```
W_internal  =  W_external(cavities)  +  W_external(robin springs)
```

where:
- **W_internal** = `integral 0.5*(S_prev + S_cur) : (E_cur - E_prev) dV` (volume integral of stress power)
- **W_external(cavities)** = `P * DeltaV` (pressure work on LV/RV endocardial surfaces)
- **W_external(robin)** = work done by the boundary springs on the epicardium and base

When the project began, the observed ratio was:

```
W_internal ~ 0.32 * W_external
```

This 70% discrepancy was investigated over ~6 weeks. The root causes turned out to be a combination of **four independent bugs**, each contributing to the mismatch. They are presented here in the order they were discovered.

---

## 2. Timeline of Bugs and Fixes

### Phase 1: Function Space Errors (Jan 20 - Feb 6)

This was the longest and most confusing phase, involving multiple failed attempts before the correct solution was found.

#### 2.1 The Starting Point (Jan 20, commit `dba784a`)

The original `MetricsCalculator` used a straightforward approach:

```python
# Create a DG1 tensor space
W_tensor = dolfinx.fem.functionspace(self.mesh, ("DG", 1, (3, 3)))

# Interpolate S and E into DG1 Functions
S_cur = dolfinx.fem.Function(W_tensor)
S_cur.interpolate(dolfinx.fem.Expression(S_ufl, W_tensor.element.interpolation_points))

# Compute work using the interpolated Functions
W_density = ufl.inner(0.5 * (S_prev + S_cur), E_cur - E_prev)
work = assemble_scalar(form(W_density * dx))
```

**The problem**: This code created a *new* `W_tensor` function space every time `update_state()` or `_calculate_true_work()` was called. On some FEniCSx/basix versions, the `("DG", 1, (3, 3))` tensor space triggered **JIT compilation crashes** — the generated C code for tabulating basis functions on a rank-2 tensor element would fail or produce incorrect results.

Additionally, the DG1 interpolation was recreating `dolfinx.fem.Function` objects at every step rather than reusing persistent storage. This made the code both slow and fragile.

**The stress source was also wrong**: The code used a separate `material_dg` object for stress computation, which was a copy of the material model. This shadow copy could diverge from the actual solver's material model, giving different stress values.

#### 2.2 Grand Unification + Flattened DG0 Vectors (Jan 21-22, commits `369b27e`, `27b3204`)

Two problems were addressed simultaneously:

**Fix A — Single stress source**: The separate `material_dg` shadow model was removed. All stress computations now used `cardiac_model.S(C)`, the same model object used by the solver. This guaranteed that the S tensor in the metrics matched the S tensor that the solver was equilibrating.

**Attempted Fix B — Flatten tensors to avoid JIT crash**: To work around the tensor-space JIT bug, the 3x3 stress and strain tensors were manually flattened into 9-component DG0 vectors:

```python
# Flatten 3x3 tensor into length-9 vector
vector_elem = basix.ufl.element("DG", cell_name, 0, shape=(9,))
W_flat = dolfinx.fem.functionspace(self.mesh, vector_elem)

def _flatten_tensor(self, T):
    return ufl.as_vector([T[i, j] for i in range(3) for j in range(3)])
```

The interpolation used a full L2 projection (assembling mass matrix + RHS, solving the linear system) to project the UFL tensor expression into this flattened vector space:

```python
# L2 projection for DG0: solve M * x = b
test = ufl.TestFunction(W_flat)
L = fem.form(ufl.inner(tensor_flat_ufl, test) * dx)
b = fem.petsc.assemble_vector(L)
A = fem.petsc.assemble_matrix(a)
target.x.array[:] = b.array / A.getDiagonal().array  # DG0: M is diagonal
```

The work integral then used `ufl.dot()` on the flattened vectors (equivalent to `ufl.inner()` on the original tensors):

```python
W_density = ufl.dot(dS_avg_flat, dE_flat)  # dot product of 9-vectors = S:dE
```

**Result**: The JIT crash was avoided, but the work values were still too low. The DG0 projection acts as a **spatial averaging filter** — it replaces the stress field within each element by its cell-average. For the nonlinear stress constitutive law (exponential Fung-type), this smoothing systematically underestimates the stress in high-gradient regions.

#### 2.3 Supervisor's DG0 Tensor Space (Jan 22, commit `f7afe26`)

The supervisor suggested using a proper DG0 tensor space instead of the flattened vector:

```python
W_flat = dolfinx.fem.functionspace(self.mesh, ('DG', 0, (3, 3)))
```

with `ufl.inner()` for the tensor contraction. This was cleaner code but didn't change the fundamental problem: DG0 still averages away intra-element stress gradients.

#### 2.4 The "Nuclear Option" — DG1 Component-wise Scalars (Jan 24, commit `3e23792`)

To get DG1 precision without the tensor-space JIT crash, each tensor component was stored as a separate scalar DG1 function:

```python
W_scalar = dolfinx.fem.functionspace(self.mesh, ("DG", 1))

# 9 separate scalar functions instead of one tensor function
S_prev_comps = [dolfinx.fem.Function(W_scalar) for _ in range(9)]
E_prev_comps = [dolfinx.fem.Function(W_scalar) for _ in range(9)]

# 9 separate Expression objects, one per component
for i in range(3):
    for j in range(3):
        expr = dolfinx.fem.Expression(S_ufl[i, j], points)
        S_comps[i*3+j].interpolate(expr)
```

The work density was then reconstructed as a sum of 9 scalar products:

```python
W_density = sum(
    0.5 * (S_prev_comps[k] + S_cur_comps[k]) * (E_cur_comps[k] - E_prev_comps[k])
    for k in range(9)
)
```

**Result**: This correctly captured intra-element gradients (DG1 matches the polynomial degree of the strain field for P2 displacement elements, since F = grad(u) + I is P1-discontinuous). The work magnitudes improved. But the code was complex (9 separate interpolations per tensor, 27+ Function objects) and the energy balance still didn't close.

**The key insight that was missed**: The problem wasn't just the function space degree — it was the fact that interpolating S into *any* finite element space and then integrating `ufl.inner(S_interpolated, dE)` introduces **projection error**. The nonlinear S(C) evaluated at quadrature points differs from S interpolated into DG1 and re-evaluated at quadrature points.

#### 2.5 The Breakthrough: Direct UFL Integration (Feb 6, commit `567d84e`)

The fix was to **stop interpolating S for the work integral entirely**. Instead of:

```python
# OLD: Interpolate S into a Function, then integrate the Function
S_func.interpolate(Expression(S_ufl, points))
work = assemble(inner(S_func, dE) * dx)
```

the code was changed to:

```python
# NEW: Integrate the raw UFL expression directly
work = assemble(inner(S_ufl, dE) * dx)
```

where `S_ufl = cardiac_model.S(C)` is the symbolic UFL expression. When FEniCSx assembles this form, it evaluates `S(C)` exactly at each quadrature point — no interpolation, no projection, no smoothing. The stress is computed from the displacement field at the precise locations where the integral is evaluated.

The strain increment `dE = E_cur - E_prev` still used interpolated Functions (since E_prev must be stored from the previous timestep), but E is a polynomial function of the displacement, so interpolation into a matched DG space is exact.

For the trapezoidal rule, the previous stress `S_prev` was still stored as an interpolated Function (unavoidable — you need to save the previous state). The formula became:

```python
wd_total = 0.5 * ufl.inner(S_ufl + S_prev_function, dE)
```

This is a **hybrid**: the current stress `S_ufl` is evaluated exactly at quadrature points, while the previous stress `S_prev_function` is the interpolated snapshot. This is the best possible approximation — exact for the current state, interpolated for the previous.

**The function space was kept configurable** via `metrics_space_type=("DG", 1)` for the stored Functions (S_prev, E_prev, E_cur), but the work integral itself always used the raw UFL expression for the current stress.

---

### Phase 2: Unit Conversion Errors (Jan 25, commits `d6d5693`, `f25ad2b`)

While debugging the function space issues, two unit conversion bugs were found in the external work calculation:

1. **1000x multiplier on internal work**: A leftover scaling factor from an earlier unit system was being applied to `W_internal`, inflating it by 3 orders of magnitude.

2. **Proxy work dV units**: The PV-loop proxy `P * dV` was using volumes in mL while pressures were in Pa, giving results in `Pa * mL` instead of `Pa * m^3` (Joules). The fix was to convert volumes to m^3 before computing work.

These were straightforward to fix once identified, but they had been masked by the much larger function-space error.

---

### Phase 3: Robin Boundary Work Formula (Jan 25 - Mar 5)

#### 3.1 The Original Robin Work (Jan 25, commit `3716601`)

The first implementation of Robin spring work used:

```python
term_epi = alpha_epi * ufl.dot(u_avg, Du) * ds(epi_marker)
```

This computes `alpha * (u . du)` — the spring resists displacement in **all directions** (fiber, sheet, normal, tangential to the surface). The integration is over the reference surface element `ds`.

#### 3.2 What the Solver Actually Does

The pulse library's `_robin_form` (in `problem.py:281-315`) implements a geometrically nonlinear Robin BC with **normal-only** resistance:

```python
# Nanson formula: map reference normal to current normal
N = self.geometry.facet_normal        # Reference normal
F = ufl.grad(u) + ufl.Identity(3)
J = ufl.det(F)
cof = J * ufl.inv(F).T               # Cofactor of F
cofN = ufl.dot(cof, N)               # Mapped normal (unnormalized)
cofnorm = ufl.sqrt(ufl.dot(cofN, cofN))  # |cofN| = area ratio
NN = cofN / cofnorm                   # Unit normal in current config

# perpendicular=False (default): project onto normal direction only
nn = ufl.outer(NN, NN)               # Normal projector
value = -nn * k * u                  # Force = -k * (u.NN) * NN
form = -ufl.dot(value, u_test) * cofnorm * ds(marker)
```

Three critical differences from the old metrics code:

| Aspect | Old Metrics | Solver (pulse) |
|--------|-------------|-----------------|
| **Direction** | All directions: `u . du` | Normal only: `(u.NN)(du.NN)` |
| **Normal vector** | Implicit reference N via `ds` | Deformed normal NN via Nanson |
| **Area element** | Reference area `ds` | Mapped area `cofnorm * ds` |

#### 3.3 The Fix (current code)

The Robin work in `_calculate_robin_work()` was rewritten to exactly mirror the solver:

```python
# Nanson formula (identical to solver)
N = ufl.FacetNormal(self.mesh)
F = ufl.grad(u_cur) + I
cof = ufl.det(F) * ufl.inv(F).T
cofN = ufl.dot(cof, N)
cofnorm = ufl.sqrt(ufl.inner(cofN, cofN))
NN = cofN / cofnorm

# Normal projections only
u_n = ufl.dot(u_avg, NN)
du_n = ufl.dot(du, NN)

# Work = -k * (u_avg . NN) * (du . NN) * cofnorm * ds
term = -alpha * u_n * du_n * cofnorm * ds
```

**Impact**: The old formula overestimated Robin work by including tangential displacement components that the solver doesn't actually penalize. This made `W_robin` too large, which in turn made `W_ext` appear larger than `W_int`, widening the energy gap.

---

### Phase 4: Exact Boundary Pressure Work (Feb - Mar 2026)

#### 4.1 The Linearization Error

The original pressure work used a linearized volume change:

```python
# Linearized: DeltaV ~ integral N . du ds (reference config)
work = p * ufl.dot(N, Du) * ds(marker)
```

At finite deformations, this systematically **overestimates** the actual cavity volume change. The error grows with the magnitude of the deformation — precisely the regime where energy balance matters most (peak systole).

#### 4.2 The Fix: Divergence Theorem Volumes

The current code computes the actual cavity volume at each configuration using the divergence theorem:

```python
# Exact volume via divergence theorem
V(u) = integral (-1/3) * J * dot(X + u, inv(F).T * N) ds

# Actual volume change
DeltaV = V(u_new) - V(u_old)

# Work with trapezoidal pressure
W = P_avg * DeltaV
```

This evaluates the true deformed cavity volume at both `u_new` and `u_old`, giving the exact volume change without linearization.

---

## 3. Summary of All Bugs

| # | Bug | Effect on W_int/W_ext | Fix Commit | Date |
|---|-----|----------------------|------------|------|
| 1 | **Interpolation smoothing**: S projected into DG0/DG1 before integration, losing nonlinear stress peaks | W_int too low (smoothed stress underestimates exponential material) | `567d84e` | Feb 6 |
| 2 | **Unit conversions**: 1000x multiplier on W_int; PV proxy in mixed units | W_int inflated by 1000x; W_ext proxy in wrong units | `d6d5693`, `f25ad2b` | Jan 25 |
| 3 | **Robin work formula**: Full `u.du` instead of normal-only `(u.NN)(du.NN)` with Nanson transform | W_robin too large (included non-physical tangential work) | `62ed08e` + later | Feb 8+ |
| 4 | **Boundary work linearization**: `N.du` on reference surface instead of exact DeltaV via divergence theorem | W_ext(cavities) overestimated at finite deformations | `567d84e` + later | Feb 6+ |

---

## 4. The Function Space Journey in Detail

This section expands on Bug #1, which consumed the most debugging time and involved the most failed attempts.

### 4.1 Why This Was Confusing

The solver in pulse computes the residual as:

```
R = integral S(C) : 0.5 * delta_C dx + (boundary terms)
```

where `S(C)` is never interpolated — it's evaluated symbolically at quadrature points during assembly. The solver therefore works with the **exact** nonlinear stress at every quadrature point.

The metrics calculator, by contrast, needed to **store** the stress from the previous timestep for the trapezoidal rule. This required interpolating `S(C)` into a finite element function. The question was: which function space?

### 4.2 The Attempts (Chronological)

| Attempt | Space | Method | Problem |
|---------|-------|--------|---------|
| v1 (Jan 20) | DG1 tensor `(3,3)` | Direct interpolation | JIT crash on some basix versions |
| v2 (Jan 21) | DG0 vector `(9,)` | L2 projection with mass matrix solve | Cell-averaged stress too smooth; PETSc overhead |
| v3 (Jan 22) | DG0 tensor `(3,3)` | Direct interpolation | Same smoothing as v2, slightly cleaner code |
| v4 (Jan 24) | DG1 scalars x 9 | Component-wise interpolation | Correct polynomial degree but 27+ Function objects; still interpolation error in work integral |
| v5 (Jan 24) | Quadrature deg 4 | Component-wise into quadrature elements | Exact at quadrature points but fragile; quadrature degree must match everywhere |
| **v6 (Feb 6)** | **DG1 tensor (for storage only)** | **Raw UFL for work integral; interpolated Functions only for state storage** | **Correct: zero projection error in work computation** |

### 4.3 Why DG0 Was Particularly Bad

For the exponential Fung-type material law used in cardiac mechanics:

```
Psi = C/2 * (exp(Q) - 1)
Q = b_f * E_ff^2 + b_t * (E_ss^2 + E_nn^2 + ...) + b_fs * (E_fs^2 + ...)
```

The stress `S = dPsi/dE` involves `exp(Q)`, which is highly nonlinear. Within a single element, the stress can vary by a factor of 2-5x between integration points (especially in thin structures like the septum or RV free wall).

DG0 replaces this variation with a single cell-average value. For the exponential function, the average of exp(x) is always less than exp(average(x)) by Jensen's inequality — so DG0 systematically **underestimates** the stress-strain work density, particularly in high-stress regions.

DG1 is better (it captures linear variation within each element) but still introduces error when the stress field has quadratic or higher-order variation — which it does, because F is P1-discontinuous (linear within elements) and S(F) is nonlinear.

### 4.4 Why Direct UFL Integration Is Exact

When you write:

```python
form = ufl.inner(S_ufl, dE) * dx
work = assemble_scalar(fem.form(form))
```

FEniCSx generates C code that, at each quadrature point:
1. Evaluates the displacement `u` (from its P2 representation)
2. Computes `F = grad(u) + I` (exact differentiation of P2 polynomials)
3. Computes `C = F^T F` (exact algebra)
4. Evaluates `S(C)` via the material law (exact nonlinear evaluation)
5. Evaluates `dE = E_cur - E_prev` (from their DG1 representations)
6. Computes `S(C) : dE` (exact contraction)
7. Multiplies by the quadrature weight and Jacobian determinant

There is no projection, no interpolation, no smoothing. The stress is computed from the raw displacement DOFs at every quadrature point. The only approximation is the quadrature rule itself (which is exact for polynomials up to the specified degree).

---

## 5. Lessons Learned

1. **Never interpolate what you can evaluate symbolically.** Storing S_prev as an interpolated function is unavoidable (you need state from the previous timestep), but integrating the *current* S should always use the raw UFL expression.

2. **Function space degree matters, but differently than expected.** The issue wasn't "DG0 vs DG1 vs Quadrature" — it was "interpolated vs symbolic." Once the current stress was kept symbolic, the stored S_prev in DG1 introduced only a small, acceptable error (second-order in the trapezoidal rule).

3. **Energy balance requires matching formulas between solver and post-processing.** The Robin work and boundary pressure work must use exactly the same mathematical formulation as the solver's variational form. Any simplification (dropping Nanson transforms, using reference normals, linearizing volume changes) breaks the energy balance.

4. **Unit consistency is invisible until it isn't.** The 1000x multiplier and mixed mL/m^3 units produced "reasonable-looking" numbers because cardiac work spans many orders of magnitude. Only the energy balance check (W_int vs W_ext) revealed the error.

5. **Multiple bugs can mask each other.** The function space error made W_int too low, while the unit error made it too high. The Robin formula error made W_ext too high. These partially cancelled, producing a stable-looking but wrong ratio of ~0.32 that persisted across multiple "fix" attempts.

---

## 6. Current Architecture (as of March 2026)

```
MetricsCalculator.__init__():
    - Creates W_tensor = ("DG", 1, (3,3)) for state storage
    - Creates persistent Functions: S_total, S_prev, E_cur, E_prev, etc.
    - Pre-compiles UFL expressions: S_tot_ufl, S_act_ufl, S_pas_ufl, S_cmp_ufl
    - Pre-compiles dolfinx.fem.Expression objects for interpolation

_calculate_state_variables():
    - Interpolates S, E into Functions (for storage, visualization, mean values)
    - Does NOT use these Functions for work integration

_calculate_incremental_work():
    - Uses raw UFL: S_tot_ufl (current, exact at quadrature points)
    - Uses interpolated Function: S_prev (previous timestep, DG1)
    - Trapezoidal: wd = 0.5 * inner(S_ufl + S_prev, dE)
    - dE = E_cur - E_prev (both DG1 Functions — acceptable since E is polynomial)

_calculate_robin_work():
    - Nanson transform: NN = cofN / |cofN|, matching pulse solver exactly
    - Normal projection only: (u.NN)(du.NN), matching perpendicular=False
    - Area mapping: cofnorm * ds, matching solver

_calculate_boundary_work_exact():
    - Exact cavity volume via divergence theorem at u_new and u_old
    - DeltaV = V(u_new) - V(u_old), no linearization
    - W = P_avg * DeltaV (trapezoidal pressure)

update_state():
    - Shifts current -> previous: E_prev = E_cur, S_prev = S_total, etc.
    - Saves u_prev for Robin/boundary work calculations
```
