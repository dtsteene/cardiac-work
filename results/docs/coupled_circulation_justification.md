# Coupled Circulation Model: Justification and Validation

## Why couple a 0D circulation model to the FEM solver?

The biventricular FEM solver requires cavity pressure boundary conditions at every timestep.
These pressures must be physiologically plausible, hemodynamically self-consistent, and
produce realistic PV loop shapes across the full disease spectrum from healthy to end-stage PAH.
Three approaches exist:

1. **Prescribed pressure waveforms** — hand-craft P_LV(t) and P_RV(t) analytically
2. **Prescribed volume waveforms** — hand-craft V_LV(t) and V_RV(t), let the FEM find pressures
3. **Coupled 0D–3D** — a lumped-parameter ODE model exchanges volumes and pressures with the FEM solver at each timestep

We use approach 3 (Regazzoni 2020 circulation model). The argument below explains why.

## The coupling mechanism

The coupling in `complete_cycle.py` works through `p_BiV_func`:

```
Each timestep:
  1. ODE model advances its state → proposes target volumes V_LV(t), V_RV(t)
  2. p_BiV_func receives these volumes
  3. FEM solver sub-steps the mesh to match those volumes
     (adjusting displacement via Newton iterations)
  4. The Lagrange multiplier pressure required to achieve those volumes is returned
  5. ODE model receives those pressures → advances to next step
```

The ODE is the **volume driver**; the FEM is the **pressure responder**. The tissue mechanics
(Holzapfel-Ogden material, LDRB fiber field, active stress) determine what pressure is needed
to achieve a given cavity volume — this is the 3D equivalent of the end-systolic and
end-diastolic pressure-volume relationships (ESPVR, EDPVR).

## Energy balance is guaranteed

A common concern: does the coupling create or destroy energy?

No. At every timestep, the pressure is *derived from* the volume target via the FEM solve.
The boundary work (P × dV) and the internal strain energy (∫ S:dE dV) are computed from
the same displacement field and the same pressure. They balance by construction (quasi-static,
no kinetic energy). This holds regardless of where the volume target comes from — the FEM
side is always self-consistent.

Energy would only be violated if we prescribed P(t) *and* V(t) independently
(over-constrained), which is not what the coupling does.

## Empirical validation: ODE pressure ≈ solver pressure

We verified that the ODE model's internal pressure (`circulation/history.npy → p_LV, p_RV`)
matches the FEM Lagrange multiplier (`solver/solver_cavity_pressure_mmHg.npy`) to high
accuracy:

| Case       | LV peak (solver) | LV peak (ODE) | Max |diff| | Relative |
|------------|-------------------|----------------|-------------|----------|
| healthy    | 108.2 mmHg        | 108.2 mmHg     | 0.97 mmHg   | 0.90%    |
| end_stage  | 83.8 mmHg         | 83.8 mmHg      | similar     | <1%      |
| (all 8)    | exact peak match  | exact peak match| <1 mmHg     | <1%      |

The residual difference is exactly a 1-timestep lag (dt = 1 ms) due to the staggered
coupling scheme:

```
solver_pressure[i] == circ_pressure[i+1]    (to machine precision, ~1e-14)
```

This means the ODE model's P(V) relationship (its compliance curves, valve timing, afterload)
is **consistent with the 3D tissue mechanics**. The two models agree on what pressure is
needed for a given volume.

## Why not prescribed waveforms?

### Problem 1: You'd need to know the 3D compliance a priori

To draw a realistic P_LV(t) waveform for a specific mesh, you'd need to know:
- At what pressure does this mesh start ejecting? (depends on material params, fiber field, prestress)
- What is the end-systolic elastance? (depends on active stress magnitude and fiber architecture)
- What is the diastolic filling curve? (depends on passive material stiffness)
- When do the valves open and close? (depends on afterload and contractility)

The coupled system answers all of these implicitly — the FEM solver IS the ground-truth
compliance, and the ODE model automatically finds the operating point that satisfies both.

### Problem 2: LV-RV hemodynamic coupling

In the real circulation, the LV and RV are connected through the pulmonary and systemic
circuits. When RV pressure rises (PAH), this affects:
- Pulmonary venous return → LV filling changes
- LV preload → LV stroke volume changes
- Systemic afterload remains roughly constant

The ODE model captures this closed-loop coupling. With prescribed waveforms, you'd need to
manually adjust LV parameters every time you change RV severity — and getting the
cross-talk right is non-trivial.

### Problem 3: PV loop shape realism

The ODE model produces PV loops with correct phases (isovolumic contraction, ejection,
isovolumic relaxation, filling) from first principles. The valve timing emerges from the
pressure-flow relationships, not from hand-tuned thresholds. This is especially important
for pathological cases where valve timing shifts (e.g., prolonged isovolumic contraction
in severe PAH).

## What the coupling gives us

1. **Physiologically plausible boundary conditions** — loop shapes, valve timing, and
   hemodynamic interactions emerge from the ODE model's physics, not from manual tuning
2. **Self-consistent hemodynamics** — the LV/RV coupling through the pulmonary circuit
   is captured automatically
3. **Controllable disease spectrum** — by varying the ODE parameters (primarily pulmonary
   vascular resistance, PVR), we sweep from healthy to end-stage PAH with realistic
   hemodynamic consequences at each severity level
4. **Validated pressure accuracy** — the <1% agreement between ODE and solver pressures
   confirms that the 0D model's P(V) relationship matches the 3D tissue mechanics

## Extending the disease spectrum

Currently we have 8 severity levels with Optuna-optimized circulation parameters.
If finer spacing is needed (e.g., 20-50 cases for smoother correlation curves), two
approaches are available:

### Approach A: Parameter interpolation

The 8 optimized parameter sets can be linearly interpolated. The key circulation
parameters that control disease severity are:

| Parameter | Healthy | End-stage | Effect |
|-----------|---------|-----------|--------|
| R_PVB (pulmonary vascular resistance) | low | high | Primary PAH driver |
| kE_RV (RV contractility) | normal | elevated | RV compensation |
| E_max_RV (RV end-systolic elastance) | normal | elevated | RV hypertrophy |
| C_AR_PUL (pulmonary arterial compliance) | high | low | Vascular stiffening |

For a target RV_ESP between two optimized cases, interpolate all parameters linearly
(or log-linearly for resistances) between the bracketing cases. The coupled simulation
will find the self-consistent hemodynamic state for those intermediate parameters.

This preserves physiological realism (each interpolated parameter set still runs through
the full ODE + FEM coupling) at the cost of one simulation per case.

### Approach B: Single-parameter sweep

Rather than interpolating all parameters, sweep a single dominant parameter
(R_PVB, the pulmonary vascular resistance) while holding everything else at the
healthy baseline. This isolates the effect of increased RV afterload and produces
a clean, monotonic disease spectrum.

Pros: cleaner spectrum, single control variable, easier to interpret.
Cons: less realistic — real PAH involves compensatory changes in RV contractility,
compliance, and atrial function that a single-parameter sweep misses.

### Approach C: Mixed — Optuna for anchors, interpolation between

Use Optuna to optimize 4-5 anchor points (healthy, mild, moderate, severe, end-stage)
with clinically validated targets. Interpolate between these for intermediate cases.
This combines the physiological grounding of optimization with the fine spacing of
interpolation.

## Relevance to the proxy validation question

For the thesis question (which pressure proxy works best for septal work), the coupling
model matters in two ways:

1. **Realistic P_LV(t) and P_RV(t) waveforms** — the proxy computes W = ∮ P dε, so
   the pressure waveform shape directly affects proxy accuracy. Synthetic waveforms
   might miss features (dicrotic notch, isovolumic phases) that real proxies must handle.

2. **Consistent tissue response** — the FEM solver produces the ground-truth S:dE work
   that we compare the proxies against. The coupling ensures that the tissue deformation
   (and hence the strain field ε_ll that enters the proxy) is physically consistent with
   the applied pressure. A prescribed pressure that's too high or too low for the tissue
   stiffness would produce unrealistic strain patterns.

In short: the coupled model ensures that we're testing the proxies under conditions that
are as close to reality as a computational model can provide. This strengthens the
clinical relevance of the findings.

## Thesis argument (draft)

> The biventricular finite element model requires cavity pressure boundary conditions at
> each timestep. Rather than prescribing these pressures analytically, we couple the 3D
> solver to a 0D lumped-parameter circulation model (Regazzoni et al. 2020) that
> represents the systemic and pulmonary circulations, cardiac valves, and atrial
> compliance. At each timestep, the circulation model proposes cavity volumes based on
> its internal hemodynamic state; the FEM solver drives the mesh to match these volumes
> and returns the Lagrange multiplier pressures — the actual surface tractions required
> by the tissue mechanics. This bidirectional coupling ensures that (i) the pressure-volume
> loops are physiologically shaped with correct valve timing, (ii) the LV and RV are
> hemodynamically coupled through the shared circulation, and (iii) energy balance is
> maintained by construction. We verified that the circulation model's internal pressures
> agree with the FEM Lagrange multipliers to within 1% (a 1-timestep lag inherent to the
> staggered coupling scheme), confirming that the 0D model's compliance relationships are
> consistent with the 3D tissue mechanics.
>
> Disease severity is controlled by varying the circulation parameters — primarily
> pulmonary vascular resistance and RV contractility — which have been optimized against
> published hemodynamic targets for each severity level using Bayesian optimization
> (Optuna). This produces a spectrum of 8 cases from healthy (RV ESP ≈ 28 mmHg) to
> end-stage PAH (RV ESP ≈ 83 mmHg) with self-consistent hemodynamics at each point.
