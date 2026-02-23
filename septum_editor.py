"""
Septum Tag Editor
=================
Interactive tool to manually correct septum (marker=3) tags on a BiV mesh.

Can be used standalone or imported by geometry_generator.py for pipeline integration.

Usage (standalone)
------------------
  python septum_editor.py <markers_scalar.xdmf> [--output <out.xdmf>]

Controls
--------
  P (hover over cell) → toggle septum tag on/off  (pyvista picking key)
  S                   → save current tags back to the HDF5 file (in-place or --output)
  R                   → reset all tags to the originally loaded values
  Z                   → undo last pick
  Q / close window    → quit (unsaved changes are lost unless you pressed S)

Marker values
-------------
  1 = LV free wall
  2 = RV free wall
  3 = interventricular septum  (green highlight)

Pipeline integration
--------------------
When called from geometry_generator.generate_and_load() with manual_refinement=True,
the editor opens after LDRB tagging. Close the window when done — the edited tags are
captured from editor.tags and written into markers_mt (the dolfinx MeshTags object
that flows into geometry.bp → additional_data["markers_mt"] → FEniCSx DOF assignment).
"""

import argparse
import sys
from pathlib import Path
from copy import deepcopy

import h5py
import numpy as np
import pyvista as pv


# ---------------------------------------------------------------------------
# Helpers to read / write the HDF5 produced by dolfinx's XDMF writer
# ---------------------------------------------------------------------------

def _h5_tags_path(xdmf_path: Path) -> tuple[Path, str]:
    """Return (h5_file, dataset_path) for the cell-tag Values array."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    # Find the DataItem inside the Attribute named mesh_tags / marker_scalar
    for attr in root.iter("Attribute"):
        di = attr.find("DataItem")
        if di is not None and di.text:
            ref = di.text.strip()          # e.g. "markers_scalar.h5:/MeshTags/mesh_tags/Values"
            h5file, dset = ref.split(":")
            h5path = xdmf_path.parent / h5file
            return h5path, dset
    raise RuntimeError("Could not locate tag DataItem in XDMF file.")


def _h5_mesh_paths(xdmf_path: Path) -> tuple[Path, str, str]:
    """Return (h5_file, topology_dset, geometry_dset)."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    topo_ref = geom_ref = None
    for di in root.iter("DataItem"):
        if di.text and "topology" in di.text:
            topo_ref = di.text.strip()
        if di.text and "geometry" in di.text:
            geom_ref = di.text.strip()
    if topo_ref is None or geom_ref is None:
        raise RuntimeError("Could not find topology/geometry DataItem in XDMF.")
    h5file, topo_dset = topo_ref.split(":")
    _, geom_dset = geom_ref.split(":")
    h5path = xdmf_path.parent / h5file
    return h5path, topo_dset, geom_dset


def load_mesh_and_tags(xdmf_path: Path):
    """Load connectivity, points, and cell tags from a dolfinx XDMF/H5 pair."""
    h5mesh, topo_dset, geom_dset = _h5_mesh_paths(xdmf_path)
    h5tags, tags_dset = _h5_tags_path(xdmf_path)

    with h5py.File(h5mesh, "r") as f:
        connectivity = f[topo_dset][:]   # (n_cells, 4)  tet node indices
        points = f[geom_dset][:]         # (n_points, 3)

    with h5py.File(h5tags, "r") as f:
        tags = f[tags_dset][:].flatten().astype(np.int32)

    return points, connectivity, tags


def save_tags(xdmf_path: Path, output_path: Path | None, tags: np.ndarray):
    """Write updated tags back to HDF5 (in-place or to a new copy)."""
    h5tags, tags_dset = _h5_tags_path(xdmf_path)

    if output_path is not None:
        import shutil
        # Copy the whole directory pair (xdmf + h5) so original is untouched
        out_h5 = output_path.parent / output_path.name.replace(".xdmf", ".h5")
        shutil.copy2(h5tags, out_h5)
        # Patch the copy
        target_h5 = out_h5
    else:
        target_h5 = h5tags

    with h5py.File(target_h5, "r+") as f:
        ds = f[tags_dset]
        if ds.shape == tags.reshape(ds.shape).shape:
            ds[...] = tags.reshape(ds.shape)
        else:
            ds.resize(tags.shape)
            ds[...] = tags

    print(f"[saved] tags written to {target_h5}")


# ---------------------------------------------------------------------------
# Build a pyvista UnstructuredGrid from raw tet data
# ---------------------------------------------------------------------------

def build_pyvista_mesh(points, connectivity, tags):
    """Create a pv.UnstructuredGrid coloured by cell tags."""
    n_cells = connectivity.shape[0]
    # pyvista cell array format: [4, n0, n1, n2, n3,  4, n0, ...]
    cell_arr = np.empty((n_cells, 5), dtype=np.int64)
    cell_arr[:, 0] = 4
    cell_arr[:, 1:] = connectivity
    cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cell_arr.ravel(), cell_types, points.astype(float))
    grid.cell_data["marker"] = tags.copy()
    return grid


# ---------------------------------------------------------------------------
# Main editor class
# ---------------------------------------------------------------------------

class SeptumEditor:
    SEPTUM_TAG = 3
    COLOURS = {1: "royalblue", 2: "tomato", 3: "limegreen"}
    OPACITY = {1: 0.65, 2: 0.65, 3: 0.95}

    def __init__(self, xdmf_path: Path, output_path: Path | None):
        self.xdmf_path = xdmf_path
        self.output_path = output_path

        points, connectivity, tags = load_mesh_and_tags(xdmf_path)
        self.points = points
        self.connectivity = connectivity
        self.tags = tags.copy()
        self._original_tags = tags.copy()
        self._history: list[tuple[int, int]] = []   # (cell_id, old_tag)

        self.grid = build_pyvista_mesh(points, connectivity, tags)

        # Build per-region sub-meshes for display
        self.plotter = pv.Plotter(title="Septum Tag Editor")
        self._actors = {}
        self._build_scene()
        self._register_callbacks()
        self._print_help()

    # ------------------------------------------------------------------
    def _build_scene(self):
        """Render each region as a separate surface actor."""
        self.plotter.clear()
        self._actors = {}

        for tag_val, colour in self.COLOURS.items():
            mask = self.tags == tag_val
            if not mask.any():
                continue
            sub = self.grid.extract_cells(np.where(mask)[0])
            surf = sub.extract_surface()
            actor = self.plotter.add_mesh(
                surf,
                color=colour,
                opacity=self.OPACITY[tag_val],
                show_edges=True,
                edge_color="black",
                line_width=0.5,
                pickable=(tag_val == self.SEPTUM_TAG),  # only septum is easily clickable
                name=f"region_{tag_val}",
            )
            self._actors[tag_val] = actor

        # Also add a fully-transparent pickable version of the whole mesh
        # so we can pick non-septum cells too
        full_surf = self.grid.extract_surface()
        full_surf.cell_data["marker"] = self.grid.extract_surface().cell_data.get(
            "marker", np.zeros(full_surf.n_cells)
        )
        self.plotter.add_mesh(
            full_surf,
            color="white",
            opacity=0.0,
            pickable=True,
            name="pick_target",
        )
        self._full_surf = full_surf

        # Legend
        self.plotter.add_legend(
            [["LV (1)", "royalblue"], ["RV (2)", "tomato"], ["Septum (3)", "limegreen"]],
            bcolor="white",
            border=True,
            size=(0.18, 0.12),
        )

    # ------------------------------------------------------------------
    def _register_callbacks(self):
        self.plotter.enable_cell_picking(
            callback=self._on_pick,
            show_message=True,   # show pyvista's "Press P to pick" hint
            through=False,
            style="surface",
            color="yellow",
            opacity=0.6,
        )
        self.plotter.add_key_event("s", self._on_save)
        self.plotter.add_key_event("S", self._on_save)
        self.plotter.add_key_event("r", self._on_reset)
        self.plotter.add_key_event("R", self._on_reset)
        self.plotter.add_key_event("z", self._on_undo)
        self.plotter.add_key_event("Z", self._on_undo)
        # Override pyvista built-in number keys that change the colour map
        for k in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            self.plotter.add_key_event(k, lambda: None)

    # ------------------------------------------------------------------
    def _surface_cell_to_volume_cell(self, surf_cell_id: int) -> int | None:
        """Map a surface cell id back to the volume cell id."""
        surf = self.plotter.picked_cells
        if surf is None:
            return None
        # pyvista stores 'vtkOriginalCellIds' when extracting surface
        full_surf = self.grid.extract_surface()
        orig_ids = full_surf.cell_data.get("vtkOriginalCellIds", None)
        if orig_ids is None:
            return None
        if surf_cell_id >= len(orig_ids):
            return None
        return int(orig_ids[surf_cell_id])

    # ------------------------------------------------------------------
    def _on_pick(self, picked):
        if picked is None or picked.n_cells == 0:
            return

        # Get the original volume cell index
        full_surf = self.grid.extract_surface()
        orig_ids = full_surf.cell_data.get("vtkOriginalCellIds", None)
        if orig_ids is None:
            print("[warn] vtkOriginalCellIds not available — cannot map to volume cell.")
            return

        # Find which surface cell was picked by comparing centres
        pick_centre = np.array(picked.center)
        surf_centres = full_surf.cell_centers().points
        dists = np.linalg.norm(surf_centres - pick_centre, axis=1)
        surf_cell_id = int(np.argmin(dists))
        vol_cell_id = int(orig_ids[surf_cell_id])

        old_tag = int(self.tags[vol_cell_id])
        if old_tag == self.SEPTUM_TAG:
            # Restore to whatever LDRB originally assigned (LV=1 or RV=2)
            new_tag = int(self._original_tags[vol_cell_id])
            action = "removed from"
        else:
            new_tag = self.SEPTUM_TAG
            action = "added to"

        self._history.append((vol_cell_id, old_tag))
        self.tags[vol_cell_id] = new_tag
        self.grid.cell_data["marker"] = self.tags.copy()
        self._build_scene()
        self.plotter.render()
        print(f"[pick] cell {vol_cell_id}: tag {old_tag} → {new_tag}  ({action} septum)  "
              f"[{len(self._history)} changes total]")

    # ------------------------------------------------------------------
    def _on_save(self):
        save_tags(self.xdmf_path, self.output_path, self.tags)
        n_sept = int((self.tags == self.SEPTUM_TAG).sum())
        print(f"[save] {n_sept} septum cells written.")

    # ------------------------------------------------------------------
    def _on_reset(self):
        self.tags = self._original_tags.copy()
        self.grid.cell_data["marker"] = self.tags.copy()
        self._history.clear()
        self._build_scene()
        self.plotter.render()
        print("[reset] all tags restored to original values.")

    # ------------------------------------------------------------------
    def _on_undo(self):
        if not self._history:
            print("[undo] nothing to undo.")
            return
        cell_id, old_tag = self._history.pop()
        self.tags[cell_id] = old_tag
        self.grid.cell_data["marker"] = self.tags.copy()
        self._build_scene()
        self.plotter.render()
        print(f"[undo] cell {cell_id} restored to tag {old_tag}.")

    # ------------------------------------------------------------------
    def _print_help(self):
        print("\n" + "=" * 60)
        print("  SEPTUM TAG EDITOR")
        print("=" * 60)
        print(f"  File : {self.xdmf_path}")
        print(f"  Cells: {len(self.tags)}  "
              f"(LV={int((self.tags==1).sum())}, "
              f"RV={int((self.tags==2).sum())}, "
              f"Septum={int((self.tags==3).sum())})")
        print()
        print("  P (hover over cell) →  toggle septum tag on/off")
        print("  S                 →  save")
        print("  Z                 →  undo last click")
        print("  R                 →  reset to original")
        print("  Q / close         →  quit (edits are captured on close)")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    def run(self):
        self.plotter.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive septum tag editor.")
    parser.add_argument("xdmf", type=Path, help="Path to markers_scalar.xdmf")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Save edited tags to this XDMF path (default: overwrite input H5).",
    )
    args = parser.parse_args()

    if not args.xdmf.exists():
        print(f"[error] File not found: {args.xdmf}")
        sys.exit(1)

    editor = SeptumEditor(args.xdmf, args.output)
    editor.run()


if __name__ == "__main__":
    main()
