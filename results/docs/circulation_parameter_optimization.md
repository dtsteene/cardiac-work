# Circulation Parameter Optimization for the PAH Severity Spectrum

## Overview

This document describes the calibration of the 0D lumped-parameter circulation model
(Regazzoni et al. 2020) for use as boundary conditions in coupled 3D-0D biventricular
FEM simulations. The goal: produce clinically plausible hemodynamics across a spectrum
of pulmonary arterial hypertension (PAH) severity levels, matched to the UKB mean
biventricular mesh geometry (LV EDV = 111.5 mL, RV EDV = 76.8 mL).

Result JSONs are in `data/ukb_circ_v2/optimized_regazzoni_ukb_{severity}.json`.

## Modification to the Circulation Library

### Nonlinear Diastolic EDPVR

The standard Regazzoni 2020 model uses a linear time-varying elastance:

    P(V, t) = E(t) * (V - V0),    where E(t) = EB + (EA - EB) * a(t)

This creates a fundamental coupling between end-diastolic pressure (EDP) and
end-diastolic volume (EDV): both are controlled by the single parameter EB.
For a small mesh (EDV ~ 111 mL for LV, 77 mL for RV), the optimizer cannot
simultaneously achieve low EDP (clinical range 4-12 mmHg) and correct EDV
without the passive stiffness being too high or too low.

We replaced the passive (diastolic) component with an exponential pressure-volume
relationship, following the well-established Klotz EDPVR framework:

    P(V, t) = (EA - EB) * a(t) * (V - V0)  +  (EB / kE) * (exp(kE * (V - V0)) - 1)

where kE (units: 1/mL) controls the nonlinear stiffening at high volumes.

Properties:
- For kE -> 0: reduces to the original linear model (backward compatible)
- At V near V0: P_passive ~ EB * (V - V0) (linear, low EDP)
- At V >> V0: P_passive grows exponentially (limits overfilling, controls EDV)
- dP/dV = (EA - EB)*a(t) + EB * exp(kE * (V - V0))

This is a standard formulation in cardiac mechanics literature. The exponential
EDPVR was established by Klotz et al. (2006) and is used in the CircAdapt model
(Arts et al. 2005) and numerous other lumped-parameter frameworks. It captures
the physiological stiffening that occurs as the ventricle approaches its
distensibility limit.

**Implementation**: `circulation/src/circulation/regazzoni2020.py`, method
`_make_pressure_func()`. The Jacobian was also updated for volume-dependent
dP/dV when kE > 0. Applied to LV and RV chambers only (atria remain linear).

**References**:
- Klotz S, Hay I, Dickstein ML et al. "Single-beat estimation of end-diastolic
  pressure-volume relationship: a novel method with potential for noninvasive
  application." Am J Physiol Heart Circ Physiol. 2006;291(1):H403-H412.
- Arts T, Delhaas T, Bovendeerd P et al. "Adaptation to mechanical load determines
  shape and properties of heart and circulation: the CircAdapt model." Am J Physiol
  Heart Circ Physiol. 2005;288(4):H1943-H1954.
- Regazzoni F, Salvador M, Africa PC et al. "A cardiac electromechanical model
  coupled with a lumped-parameter model for closed-loop blood circulation."
  J Comput Phys. 2022;457:111083.

## Hemodynamic Targets

### Clinical Basis

Targets were derived from the 2022 ESC/ERS Guidelines for the diagnosis and
treatment of pulmonary hypertension, supplemented by the Kovacs et al. (2009)
systematic review of normal pulmonary hemodynamics.

Key thresholds from the 2022 guidelines:
- Normal mPAP: 14.0 +/- 3.3 mmHg (Kovacs et al. 2009, n=1187)
- PAH diagnosis: mPAP > 20 mmHg AND PVR > 2 WU AND PAWP <= 15 mmHg
- Pre-capillary phenotype requires PAWP <= 15 mmHg (tracked via LA_P_MEAN)

The severity spectrum was designed to span normal hemodynamics through end-stage
RV-LV pressure equalization, which is the regime where septal transmural pressure
(P_LV - P_RV) approaches zero and the choice of pressure proxy for septal work
becomes critical.

**References**:
- Humbert M, Kovacs G, Hoeper MM et al. "2022 ESC/ERS Guidelines for the
  diagnosis and treatment of pulmonary hypertension." Eur Heart J.
  2022;43(38):3618-3731.
- Kovacs G, Berghold A, Scheidl S, Olschewski H. "Pulmonary arterial pressure
  during rest and exercise in healthy subjects: a systematic review."
  Eur Respir J. 2009;34(4):888-894.
- Simonneau G, Montani D, Celermajer DS et al. "Haemodynamic definitions and
  updated clinical classification of pulmonary hypertension." Eur Respir J.
  2019;53(1):1801913.

### Target Table

| Severity        | RV ESP | LV ESP | RV EDP | LV EDP | Clinical basis                         |
|-----------------|--------|--------|--------|--------|----------------------------------------|
| healthy         |     22 |    118 |      4 |      8 | Normal (sPAP < 25, mPAP ~ 14)         |
| borderline      |     30 |    118 |      5 |      9 | Upper normal / early PH transition     |
| mild            |     38 |    118 |      6 |      9 | Mild PAH (mPAP 21-24, PVR 2-3 WU)     |
| moderate        |     55 |    110 |      8 |      8 | Moderate PAH (mPAP 35-45, PVR 4-6 WU) |
| moderate_severe |     63 |    105 |     10 |      7 | Moderate-severe PAH                    |
| severe          |     72 |    100 |     12 |      6 | Severe PAH (mPAP > 45, PVR > 6 WU)    |
| very_severe     |     85 |     95 |     14 |      5 | Near-equalization of ventricular ESP   |
| end_stage       |     95 |     90 |     16 |      4 | End-stage, transmural P ~ 0            |

Additional constraints per case:
- Volume targets: LV EDV = 111.5 mL, RV EDV = 76.8 mL (from UKB mesh)
- EF floors: 55% (healthy) scaling down to 20% (end_stage)
- CO minimums: 4.0 L/min (healthy) scaling down to 1.5 L/min (end_stage)
- LV-RV stroke volume balance: penalized if > 5% mismatch (closed-loop constraint)
- LA_P_MEAN target: 7-9 mmHg (ensures PAWP < 15, pre-capillary phenotype)

The LV ESP targets decrease with severity to reflect ventricular interdependence:
as RV pressure rises, the septum shifts leftward, impairing LV filling and reducing
systolic pressure. This is the "D-sign" seen clinically on echocardiography and is
quantified by the LV eccentricity index (Ryan et al. 1985).

**References**:
- Ryan T, Petrovic O, Dillon JC et al. "An echocardiographic index for
  separation of right ventricular volume and pressure overload."
  J Am Coll Cardiol. 1985;5(4):918-927.
- Vonk Noordegraaf A, Westerhof BE, Westerhof N. "The relationship between the
  right ventricle and its load in pulmonary hypertension." J Am Coll Cardiol.
  2017;69(2):236-243.

## Optimization Method

### Optimizer: CMA-ES

We used Covariance Matrix Adaptation Evolution Strategy (CMA-ES) via Optuna's
`CmaEsSampler`. CMA-ES was chosen over Tree-structured Parzen Estimator (TPE)
because:

1. The 22 parameters are continuous and highly correlated (e.g., increasing
   R_AR_SYS requires adjusting C_AR_SYS to maintain MAP). TPE treats parameters
   independently and cannot learn these correlations.
2. CMA-ES explicitly models the parameter covariance matrix and adapts it
   during optimization, efficiently navigating correlated search spaces.
3. Empirically, CMA-ES converged to 2-5x lower objective values than TPE
   with the same number of trials on this problem.

**Reference**:
- Hansen N, Ostermeier A. "Completely derandomized self-adaptation in evolution
  strategies." Evol Comput. 2001;9(2):159-195.

### Warm-Starting Strategy

For severe PAH cases, random initialization placed CMA-ES in basins far from
the global optimum. The solution: warm-start from the adjacent, already-optimized
severity level. This mirrors actual disease progression — PAH worsens by increasing
pulmonary vascular resistance, with all other parameters adapting continuously.

Procedure:
1. Optimize moderate_severe first (converges easily)
2. For severe: take moderate_severe params, increase R_AR_PUL by ~50%, enqueue
   as the first CMA-ES trial
3. For very_severe: start from severe, increase R_AR_PUL further
4. For end_stage: start from very_severe

This chained approach produced dramatically better results: very_severe objective
improved from 67 (random start) to 11.4 (warm-started), a 6x improvement.

### Cost Function Design

The multi-objective cost function uses dynamic weight relaxation:

    w_effective = w_base * (rel_error / threshold)   if rel_error < threshold
    w_effective = w_base                              otherwise

with threshold = 5%. This prevents the optimizer from over-focusing on targets
that are already close, freeing it to improve the remaining gaps. Without dynamic
weights, the optimizer oscillates: fixing one target breaks another.

Cost components and base weights:
- RV/LV ESP: 200 each (defines severity classification)
- RV/LV EDP: 150 each (diastolic function)
- Ao DBP: 100 (systemic afterload)
- LA P mean: 150 (pre-capillary constraint)
- LV/RV EDV: 400 each (mesh volume matching)
- SV balance: 300 (closed-loop mass conservation)
- EF floor: 400 (one-sided barrier, no penalty above floor)
- CO minimum: 500 (one-sided barrier)
- Beat-to-beat SV drift: 200 (steady-state convergence)

### Soft Penalties Instead of Pruning

Failed trials (NaN, excessive pressure, timing violations) return graded penalty
values (5000-9000) instead of being pruned. This gives CMA-ES gradient information
about the boundary of the feasible region, improving convergence near constraints.

### Simulation Settings

- Beats per trial: 20 (increased from 10 to allow venous transient convergence;
  venous time constant tau = R_VEN * C_VEN ~ 20s at 75 bpm = 25 beats)
- Time step: dt = 1e-3 s (stable for all parameter combinations)
- Verification: best trial re-solved at 50 beats, dt = 1e-3 for final JSON export

## Parameters Optimized

22 free parameters spanning cardiac chambers, valves, and vascular network:

| Parameter             | Range         | Role                              |
|-----------------------|---------------|-----------------------------------|
| RV.EA                 | 0.3 - 2.5     | RV max elastance (contractility)  |
| RV.EB                 | 0.001 - 0.025 | RV passive elastance              |
| RV.V0                 | 10 - 140      | RV unstressed volume (mL)         |
| RV.TC, RV.TR          | timing        | RV contraction/relaxation (s)     |
| RV.kE                 | 0 - 0.04      | RV diastolic stiffening (1/mL)    |
| LV.EA                 | 0.5 - 10      | LV max elastance                  |
| LV.EB                 | 0.002 - 0.030 | LV passive elastance              |
| LV.V0                 | 20 - 80       | LV unstressed volume (mL)         |
| LV.TC, LV.TR          | timing        | LV contraction/relaxation (s)     |
| LV.kE                 | 0 - 0.04      | LV diastolic stiffening (1/mL)    |
| LA.EA, LA.EB           | elastance     | Left atrial function              |
| MV.Rmin               | resistance    | Mitral valve                      |
| SYS.R_AR, C_AR, C_VEN | R, C          | Systemic arterial/venous          |
| PUL.R_AR, C_AR        | R, C          | Pulmonary arterial                |
| PUL.R_VEN, C_VEN      | R, C          | Pulmonary venous                  |
| TOTAL_VOLUME_OFFSET   | 0 - 2000 mL   | Blood volume adjustment           |

## Achieved Hemodynamics

Final results (best of v6/v7/v8/v9 per case, evaluated at 25 beats):

| Severity        | LV ESP | RV ESP | LV EDP | RV EDP | mPAP | LV EDV | RV EDV | LV EF | RV EF |   CO | TransP |
|-----------------|--------|--------|--------|--------|------|--------|--------|-------|-------|------|--------|
| healthy         |  117.8 |   29.5 |    6.7 |    4.0 | 14.0 |  111.3 |   79.0 | 55.6% | 80.5% | 4.64 |   88.3 |
| borderline      |  117.6 |   30.4 |    7.5 |    5.0 | 15.3 |  112.2 |   84.5 | 57.4% | 78.2% | 4.83 |   87.2 |
| mild            |  117.5 |   38.2 |    7.6 |    5.9 | 22.6 |  112.5 |   76.9 | 45.9% | 67.1% | 3.87 |   79.4 |
| moderate        |  111.7 |   45.2 |    8.0 |    8.2 | 19.9 |  108.9 |   79.0 | 54.8% | 75.5% | 4.48 |   66.5 |
| moderate_severe |  111.6 |   62.5 |    7.4 |    9.5 | 34.0 |  109.4 |   75.5 | 38.7% | 56.1% | 3.18 |   49.1 |
| severe          |  100.6 |   70.8 |    6.1 |   13.7 | 40.9 |  112.1 |   77.3 | 39.6% | 57.8% | 3.33 |   29.8 |
| very_severe     |   95.6 |   85.0 |    4.9 |   13.9 | 40.4 |  110.0 |   77.1 | 42.0% | 60.0% | 3.47 |   10.6 |
| end_stage       |   91.1 |   88.3 |    4.0 |   16.0 | 66.7 |  111.6 |   78.7 | 35.2% | 47.3% | 2.72 |    2.8 |

Note: healthy and moderate are being re-optimized with revised targets (v9) and
will be updated when complete. Values above are from v7/v6 respectively.

Clinical trend correlations:
- Transmural pressure vs RV ESP: r = -0.989 (near-perfect collapse)
- LV ESP vs RV ESP: r = -0.877 (ventricular interdependence)
- CO vs RV ESP: r = -0.837 (forward failure progression)

## Version History

| Version | Sampler | dt    | Beats | Key change                              |
|---------|---------|-------|-------|-----------------------------------------|
| v1-v3   | TPE     | 5e-3  | 10    | Initial attempts; dt too large, all NaN |
| v4      | TPE     | 1e-3  | 10    | First working results; LV EF too low    |
| v5      | TPE     | 1e-3  | 10    | Added kE; search space too large        |
| v6      | TPE     | 1e-3  | 10    | Capped kE, increased volume weight      |
| v7      | CMA-ES  | 1e-3  | 20    | Soft penalties, dynamic weights, 20 beats |
| v8      | CMA-ES  | 1e-3  | 20    | Warm-starting from adjacent severity    |
| v9      | CMA-ES  | 1e-3  | 20    | Revised healthy/moderate targets        |

## File Locations

- Optimizer script: `/global/D1/homes/dtsteene/circulation/examples/optimize_mesh_circ.py`
- Modified circulation library: `/home/dtsteene/D1/circulation/` (pip install -e)
- Result JSONs: `/global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v2/`
- Optuna databases: `/global/D1/homes/dtsteene/circulation/examples/results_mesh_ukb/`
- Slurm submission: `/global/D1/homes/dtsteene/cardiac-work/run_circ_optim.sbatch`
- Backup of v6 PAH results: `/global/D1/homes/dtsteene/cardiac-work/data/ukb_circ_v6_backup/`
