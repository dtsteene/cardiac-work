# Passive-softening pilot

**Status:** canonical (current). `sims/2026-07-08/softmat_pilot_L10/`
**Reported in:** `results/handover/supervisor_2026-08/story1_softening/`

## Why it was run

This pilot answers a specific challenge rather than exploring a space. Across
the pulmonary afterload sweep, the LV free-wall and septal fibre stress–strain
loops barely change between cases, and their end-diastolic points — the loop
tips — essentially coincide. Espen raised this: he expected higher afterload to
push the LV ED point outward, as it does in the CircAdapt (TriSeg) simulations
in Lee et al. 2025.

The hypothesis tested here was that the passive material was too stiff, so that
the same filling pressure could not reach a large enough end-diastolic stretch
to separate the cases.

## Design

The Holzapfel–Ogden moduli were scaled to ×1.0, ×0.5 and ×0.33 (`scale100`,
`scale050`, `scale033`), and the mildest and most severe cases — `case0_rv25`
and `case7_rv95` — were re-run at each stiffness. Six runs, at h = 10 mm since
this is a pilot.

## What it showed

Softening does what softening should: the loops become lower and longer, because
softer tissue reaches more stretch for the same pressure. But **the
case-to-case separation does not open**. Within each stiffness the baseline and
severe ED tips stay together in the LV and the septum, while the RV separates
regardless. The LV ED fibre strain of the two cases stays parallel with a
near-constant gap, and the LV stroke-work gap stays flat at roughly 17–19%,
saturating between ×0.5 and ×0.33.

So softening changes loop *magnitude*, not case-to-case *separation* at
end-diastole. The hypothesis is disconfirmed, and the reason is
[physics rather than a bug](../findings/ed-overlap.md).

## Why this matters more than a null result usually does

A negative pilot would ordinarily be a footnote. This one is load-bearing
because it closes off the most plausible "your model is wrong" objection to the
main result. Had softening recovered the gap, the flat LV ED would have been a
material-calibration artifact and several downstream conclusions would have been
in question. It did not, and the RV — through the identical mesh, solver,
coupling and postprocessing — separates cleanly in every configuration tested.
That contrast is what turns "the LV does not move" from a suspected bug into an
observation about diastolic mechanics.
