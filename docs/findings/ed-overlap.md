# The flat LV end-diastolic point is physics, not a bug

Source: `results/handover/supervisor_2026-08/story1_softening/` and the
[softening pilot](../experiments/softening-pilot.md).

## The observation

Across the pulmonary afterload sweep the LV free-wall and septal fibre
stress–strain loops barely change between cases, and their end-diastolic points
— the loop tips — essentially coincide. Espen raised this as a problem: higher
afterload should push the LV ED point outward, as it does in the CircAdapt
(TriSeg) simulations in Lee et al. 2025.

The RV, in the same figures, behaves entirely differently: its loops fan out and
its ED points march from +0.02 to +0.05.

## Why it is not a modelling error

Five arguments, in descending order of force.

**The RV is a positive control, and it is decisive.** The RV free wall goes
through the *identical* mesh, solver, coupling and postprocessing, and it
separates cleanly. A bug that clamped end-diastole would clamp it everywhere. It
does not.

**End-diastolic volumes are not clamped either.** FEM end-diastolic volume moves
monotonically across the sweep — LV 111 → 103 mL (−7%), RV 77 → 99 mL (+29%) —
and tracks the 0D warm-up faithfully. The preload reaching the mesh is real and
case-dependent.

**The magnitudes are exactly what geometry predicts.** A −7% volume change is
about −2.4% in linear dimension, or roughly 0.4 strain-points. That is precisely
the tiny ED-strain spread observed. It is the cube-root relation between volume
and linear strain, not a clamp. This is also why pressure–volume and
pressure–strain are not interchangeable: the same ED spread is plainly visible
on the volume axis and compressed into a sliver on the strain axis.

**End-diastole is a passive state.** At ED active tension is approximately zero,
so ED strain is set by passive inflation to the target volume against the shared
passive material. Same material plus nearly the same target volume gives nearly
the same ED strain, deterministically.

**It is not a missing septum.** The model does include a shared,
mechanically-coupled septum: the coupled solve drives a real biventricular mesh
and returns both cavity pressures, so ventricular interdependence is fully in
the loop. What limits the ED shift is that **interdependence scales with the
transseptal gradient**, and at end-diastole both cavities sit near their filling
pressures (`P_LV ≈ 7`, `P_RV ≈ 5–10` mmHg). That gradient is small, so the
septum barely bows at ED. The D-sign is a *systolic* event — visible in the
loading-sweep animations, where the septum bows into the LV in systole while
sitting nearly flat at end-diastole.

## The test that could have overturned it

The [softening pilot](../experiments/softening-pilot.md) is what makes this an
argument rather than an assertion. If the passive material had simply been too
stiff, softening it should have separated the ED points. Scaling the
Holzapfel–Ogden moduli to ×0.5 and ×0.33 lowered and lengthened the loops, as
expected, but left the case-to-case separation unchanged: the LV ED fibre
strains stayed parallel and the LV stroke-work gap stayed flat at ~17–19%,
saturating between ×0.5 and ×0.33.

Softening changes loop magnitude, not case separation. The hypothesis was
tested and failed.

## What remains worth checking

Before treating this as a deficiency relative to TriSeg, it is worth confirming
that CircAdapt's *pulmonary-banding* LV ED point actually shifts more than ours.
In the Lee paper the large LV changes appear in the aortic (double-banding) arm,
so the expectation may be calibrated to that case rather than to pulmonary
loading alone. That comparison has not been done.
