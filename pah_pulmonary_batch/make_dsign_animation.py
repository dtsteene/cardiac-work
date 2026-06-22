#!/usr/bin/env python3
"""
Septal D-sign work-density animation across the PAH sweep (pedagogical).

Reads the ED-static export (paraview_exports/pah_pulmonary_ed/<bundle>/ed_meshes/),
takes a short-axis clip (perpendicular to the PCA long axis) of each severity case,
colours it by work density, and assembles a severity-sweep GIF showing the RV dilating
and the septum flattening into the LV (D-sign), with where myocardial work concentrates.

Run:  python3 pah_pulmonary_batch/make_dsign_animation.py [--bundle no_frank_starling]
                                                          [--scalar w_total_density_Pa]
"""
import argparse, glob
from pathlib import Path
import numpy as np
import pyvista as pv
from PIL import Image

pv.OFF_SCREEN = True

CASES = ["case0_rv25", "case1_rv35", "case2_rv45", "case3_rv55",
         "case4_rv65", "case5_rv75", "case6_rv85", "case7_rv95"]
RV_SYS = [25, 35, 45, 55, 65, 75, 85, 95]


def apicobasal_axis(g):
    """Apicobasal direction = the largest geometric extent PERPENDICULAR to the
    LV->RV (transverse) line. Slicing perpendicular to it gives a short-axis view
    where both cavities and the septum between them are visible (the D-sign view)."""
    cent = g.cell_centers().points
    tag = np.asarray(g.cell_data["region_tag"])
    lv_c = cent[tag == 1].mean(0)
    rv_c = cent[tag == 2].mean(0)
    t = rv_c - lv_c
    t /= np.linalg.norm(t)                      # transverse (LV->RV)
    P = g.points - g.points.mean(0)
    P_perp = P - np.outer(P @ t, t)             # drop the transverse component
    w, V = np.linalg.eigh(np.cov(P_perp.T))
    a = V[:, np.argmax(w)]                       # apicobasal (tallest perp to transverse)
    return g.points.mean(0), a / np.linalg.norm(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="no_frank_starling")
    ap.add_argument("--scalar", default="w_total_density_Pa")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ed = repo / f"paraview_exports/pah_pulmonary_ed/{args.bundle}/ed_meshes"
    out = repo / f"results/handover/pah_pulmonary_paper_20260611/{args.bundle}/presentation"
    out.mkdir(parents=True, exist_ok=True)
    files = [ed / f"{c}_ed.vtu" for c in CASES]
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit(f"no ED VTUs under {ed} — run the ED-static export first")

    # global color scale: robust percentiles of |scalar| across all cases (work shown positive)
    vals = []
    for f in files:
        g = pv.read(str(f))
        vals.append(np.abs(np.asarray(g.cell_data[args.scalar])))
    allv = np.concatenate(vals); allv = allv[np.isfinite(allv)]
    clim = (0.0, float(np.percentile(allv, 98)))

    # fixed geometry frame from the most severe case (largest RV) so camera/clip are stable
    gref = pv.read(str(files[-1]))
    center, axis = apicobasal_axis(gref)
    span = np.ptp(gref.points, axis=0).max()
    cam_pos = center + axis * span * 2.2
    up = np.array([0, 0, 1.0])
    if abs(np.dot(up, axis)) > 0.9:
        up = np.array([0, 1.0, 0])

    frames = []
    for f, c, rv in zip(files, CASES, RV_SYS):
        g = pv.read(str(f))
        g.cell_data["work_pos"] = np.abs(np.asarray(g.cell_data[args.scalar]))
        # short-axis SLICE (planar cut perpendicular to the long axis), taken at
        # mid-ventricle (slightly toward base) where the D-sign is clearest.
        origin = center + axis * (span * 0.10)
        sl = g.slice(normal=axis, origin=origin)
        p = pv.Plotter(off_screen=True, window_size=(760, 760))
        p.add_mesh(sl, scalars="work_pos", clim=clim, cmap="inferno",
                   show_edges=False, scalar_bar_args=dict(title="work density (Pa)", n_labels=4))
        p.camera_position = [tuple(cam_pos), tuple(center), tuple(up)]
        p.add_text(f"RV systolic ~{rv} mmHg", position="upper_left", font_size=14, color="black")
        img = p.screenshot(return_img=True)
        p.close()
        frames.append(Image.fromarray(img))

    gif = out / "dsign_worksweep.gif"
    # ping-pong + hold so the D-sign progression reads in a loop
    seq = frames + frames[::-1]
    seq[0].save(str(gif), save_all=True, append_images=seq[1:], duration=500, loop=0)
    # also dump the per-severity frames as a static contact sheet for slides
    cols = len(frames); w, h = frames[0].size
    sheet = Image.new("RGB", (w * cols, h), "white")
    for i, fr in enumerate(frames):
        sheet.paste(fr, (i * w, 0))
    sheet.save(str(out / "dsign_worksweep_contactsheet.png"))
    print(f"[{args.bundle}] wrote {gif.name} ({len(frames)} frames) + contact sheet -> {out}")


if __name__ == "__main__":
    main()
