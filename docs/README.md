# Project knowledge base

What was run, what it showed, and what is still open. Written for a supervisor
or collaborator picking this project up without having lived through it.

The repo [README](../README.md) explains how to *run* the pipeline; this tree
explains *why* the runs exist and what they mean. [HANDOVER.md](../HANDOVER.md)
covers environment and cluster setup, [WORKFLOW.md](../WORKFLOW.md) the
end-to-end pipeline.

## Start here

If you read three pages, read these:

1. **[Open questions](open-questions.md)** — what is unresolved, and one stale
   document that still repeats a superseded conclusion. Read this before
   quoting any septal number.
2. **[Data provenance](provenance.md)** — where results live and, importantly,
   which raw simulation data no longer exists.
3. **[Findings](findings/README.md)** — the scientific conclusions, region by
   region.

## Layout

```
docs/
├── README.md              this page
├── open-questions.md      unresolved issues and known gaps
├── provenance.md          where data lives; what was deleted; reproducibility
├── experiments/           what was run
│   ├── README.md          registry: canonical vs superseded vs dead
│   ├── thesis-capped-sweep.md
│   ├── pulmonary-afterload-sweep.md
│   ├── softening-pilot.md
│   └── supporting-studies.md
└── findings/              what it showed
    ├── README.md
    ├── septal-proxy.md
    ├── rv-proxy.md
    └── ed-overlap.md
```

## The question, in one paragraph

Clinicians estimate regional myocardial work from pressure-strain loops,
`W ≈ ∮ P dε`. For the left and right free walls the choice of `P` is obvious —
the pressure in the cavity that wall encloses. The septum is the problem: it is
a shared wall, loaded by `P_LV` on one side and `P_RV` on the other, and in
pulmonary arterial hypertension `P_RV` rises toward `P_LV` until the septum
flattens. So which pressure should a clinician use for the septum? We answer it
by simulating biventricular mechanics, computing ground-truth internal work
`W = ∫∫ S : dE dV` from the stress and strain tensors, and asking which
pressure-strain proxy tracks that truth.

## A caution about correlation

The pulmonary sweep raises RV afterload monotonically with LV loading held
fixed. On that design `P_LV` is nearly constant across cases, so `P_RV`, `Mean`,
`Sum` and transmural `P_LV − P_RV` are all affine functions of `P_RV` alone and
are therefore **mathematically indistinguishable by Pearson correlation**.
Several proxies will report r ≈ 1 while disagreeing badly about how much work
is actually done. Magnitude-preserving views (indexed tracking, ratio
preservation) are the honest test. This is the motivation for the unbuilt
[RV × LV afterload grid](open-questions.md#rv--lv-afterload-grid).
