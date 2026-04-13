# Partition Notes

## Blacklisted

- **rome16q** — recurring shared-filesystem I/O errors. Jobs fail mid-run with "Input/output error" on echo/write and "process manager error waiting for completion". Has happened multiple times (2026-04-11). DO NOT USE for any simulation or postprocessing jobs.

## Reliable

- **habanaq** (h001) — 1 node, 144 cores. Reliable for all workloads. Main limitation: only 1 node, so contention when many jobs run simultaneously.
- **mi50q** (n001-n003) — 3 nodes. Reliable. Sometimes fully allocated.

## Untested / Unknown

- **slowq** (n041-n048) — 8 nodes. Slow to start, but our earlier circ_opt jobs ran there without I/O issues. OK for non-time-critical work.
- **defq** (n001-n004) — default partition, mixed availability.
