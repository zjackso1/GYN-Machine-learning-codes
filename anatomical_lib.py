"""
Straight needle library + visualization for BrachyRL.
=================================================

This version includes **ANATOMICAL cross-section sampling** of the vagina
at the chosen entry plane, ensuring needles use the FULL vaginal diameter.

- structure_mask is indexed as (Z,Y,X)
- voxel_spacing = (dz, dy, dx)
- world coords come from the full NIfTI affine
"""


from __future__ import annotations
import argparse
import os
import sys
import json
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import pyvista as pv
import nibabel as nib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.rt_brachy_env import BrachyRL_TG43
from env.structure_utils import load_structure_cache

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None



# ============================================================
#  PCA + BASIS HELPERS
# ============================================================

def _pca_major_axis(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    return axis / (np.linalg.norm(axis) + 1e-8)


# ============================================================
#  RTPLAN APPLICATOR HELPERS
# ============================================================

def _find_rtplan(rtstruct_path: str | None) -> str | None:
    if not rtstruct_path:
        return None
    struct_dir = os.path.dirname(rtstruct_path)
    if not struct_dir or not os.path.isdir(struct_dir):
        return None
    candidates = sorted(
        p for p in os.listdir(struct_dir)
        if p.startswith("RP") and p.lower().endswith(".dcm")
    )
    if not candidates:
        return None
    return os.path.join(struct_dir, candidates[0])


def _collect_rtplan_channels(ds) -> list:
    channels = []
    visited = set()

    def visit(dataset):
        for elem in dataset:
            if elem.VR != "SQ":
                continue
            if elem.keyword == "ChannelSequence":
                for item in elem.value:
                    if id(item) not in visited:
                        visited.add(id(item))
                        channels.append(item)
            for item in elem.value:
                visit(item)

    visit(ds)
    return channels


def _extract_channel_points(channel) -> np.ndarray:
    seq = None
    if hasattr(channel, "BrachyControlPointSequence"):
        seq = channel.BrachyControlPointSequence
    elif hasattr(channel, "ControlPointSequence"):
        seq = channel.ControlPointSequence
    if not seq:
        return np.zeros((0, 3), dtype=np.float32)

    points = []
    for cp in seq:
        pos = getattr(cp, "ControlPoint3DPosition", None)
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        if arr.size != 3:
            continue
        if not points or np.linalg.norm(arr - points[-1]) > 1e-3:
            points.append(arr)
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(points, axis=0)


def _channel_label(channel) -> str:
    for key in ("ChannelName", "ChannelID", "ChannelDescription", "SourceApplicatorID"):
        val = getattr(channel, key, None)
        if val:
            return str(val)
    return "Unnamed"


def _match_channels(infos, keywords):
    if not keywords:
        return []
    needles = [s.lower() for s in keywords]
    selected = []
    for info in infos:
        for key in ("name", "id", "description", "source"):
            val = info.get(key)
            if not val:
                continue
            text = str(val).lower()
            if any(n in text for n in needles):
                selected.append(info)
                break
    # de-dup by index
    seen = set()
    unique = []
    for info in selected:
        idx = info["index"]
        if idx in seen:
            continue
        seen.add(idx)
        unique.append(info)
    return unique


def _filter_inbounds(points_zyx: np.ndarray, shape):
    if points_zyx.size == 0:
        return points_zyx
    z = points_zyx[:, 0]
    y = points_zyx[:, 1]
    x = points_zyx[:, 2]
    keep = (
        (z >= 0) & (z < shape[0]) &
        (y >= 0) & (y < shape[1]) &
        (x >= 0) & (x < shape[2])
    )
    if not np.all(keep):
        points_zyx = points_zyx[keep]
    return points_zyx


def _load_ct_geometry(ct_series_path: str):
    if pydicom is None:
        raise RuntimeError("pydicom is required to load CT geometry.")
    ct_files = []
    for f in os.listdir(ct_series_path):
        if f.startswith("."):
            continue
        path = os.path.join(ct_series_path, f)
        if os.path.isfile(path):
            ct_files.append(path)
    if not ct_files:
        raise ValueError("No CT files found for geometry.")

    rows = []
    for path in ct_files:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Modality"],
        )
        if getattr(ds, "Modality", None) != "CT":
            continue
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        ps = np.array(ds.PixelSpacing, dtype=float)
        rows.append((ipp, iop, ps))

    if not rows:
        raise ValueError("No CT slices with ImagePositionPatient found.")

    ipp0, iop0, ps0 = rows[0]
    row_dir = iop0[:3]
    col_dir = iop0[3:]
    normal = np.cross(row_dir, col_dir)
    proj = [float(ipp.dot(normal)) for ipp, _iop, _ps in rows]
    idx = int(np.argmin(proj))
    origin_ipp = rows[idx][0]
    if len(proj) > 1:
        proj_sorted = np.sort(np.array(proj))
        slice_spacing = float(np.median(np.diff(proj_sorted)))
    else:
        slice_spacing = 1.0

    return {
        "origin_ipp": origin_ipp,
        "row_dir": row_dir,
        "col_dir": col_dir,
        "normal": normal,
        "row_spacing": float(ps0[0]),
        "col_spacing": float(ps0[1]),
        "slice_spacing": float(slice_spacing),
    }


def _rtplan_points_to_vox_zyx(points_mm: np.ndarray, ct_geom: dict) -> np.ndarray:
    if points_mm.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    origin_ipp = ct_geom["origin_ipp"]
    row_dir = ct_geom["row_dir"]
    col_dir = ct_geom["col_dir"]
    normal = ct_geom["normal"]
    row_spacing = ct_geom["row_spacing"]
    col_spacing = ct_geom["col_spacing"]
    slice_spacing = ct_geom["slice_spacing"]

    v = points_mm - origin_ipp[None, :]
    i = (v @ row_dir) / max(row_spacing, 1e-6)
    j = (v @ col_dir) / max(col_spacing, 1e-6)
    k = (v @ normal) / max(slice_spacing, 1e-6)
    vox = np.stack([k, i, j], axis=1).astype(np.float32)  # z,y,x
    return vox


def _load_rtplan_applicators(rtplan_path, ct_series, structure_mask):
    if pydicom is None:
        raise RuntimeError("pydicom is required to load RTPLAN applicators.")
    ct_geom = _load_ct_geometry(ct_series)

    ds = pydicom.dcmread(rtplan_path, stop_before_pixels=True)
    channels = _collect_rtplan_channels(ds)
    if not channels:
        raise RuntimeError("No ChannelSequence found in RTPLAN.")

    infos = []
    for idx, ch in enumerate(channels):
        infos.append({
            "index": idx,
            "number": getattr(ch, "ChannelNumber", None),
            "name": getattr(ch, "ChannelName", None),
            "id": getattr(ch, "ChannelID", None),
            "description": getattr(ch, "ChannelDescription", None),
            "source": getattr(ch, "SourceApplicatorID", None),
            "channel": ch,
        })

    tandem_matches = _match_channels(
        infos,
        ["tandem", "applicator1", "applicator 1", "applicator-1"],
    )
    if len(tandem_matches) != 1:
        labels = [f"[{m['index']}] {_channel_label(m['channel'])}" for m in tandem_matches]
        raise RuntimeError(
            "Could not uniquely identify tandem channel in RTPLAN. "
            f"Matches={labels} (expected exactly 1)."
        )

    ovoid_matches = _match_channels(
        infos,
        ["ovoid", "applicator2", "applicator 2", "applicator-2",
         "applicator3", "applicator 3", "applicator-3"],
    )

    tandem_info = tandem_matches[0]
    tandem_pos_mm = _extract_channel_points(tandem_info["channel"])
    tandem_pos_zyx = _filter_inbounds(_rtplan_points_to_vox_zyx(tandem_pos_mm, ct_geom), structure_mask.shape)

    ovoid_paths_zyx = []
    for info in ovoid_matches:
        pos_mm = _extract_channel_points(info["channel"])
        pos_zyx = _filter_inbounds(_rtplan_points_to_vox_zyx(pos_mm, ct_geom), structure_mask.shape)
        if pos_zyx.size:
            ovoid_paths_zyx.append(pos_zyx)

    return tandem_pos_zyx, ovoid_paths_zyx


def _make_orthonormal_basis(w: np.ndarray):
    w = w / (np.linalg.norm(w) + 1e-8)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, tmp); u /= (np.linalg.norm(u) + 1e-8)
    v = np.cross(w, u);   v /= (np.linalg.norm(v) + 1e-8)
    return u, v, w


def _shift_mask(mask: np.ndarray, shift_zyx: np.ndarray) -> np.ndarray:
    shift = np.asarray(shift_zyx, dtype=int)
    if shift.shape[0] != 3:
        raise ValueError("shift_zyx must be length-3 (z,y,x)")
    dz, dy, dx = int(shift[0]), int(shift[1]), int(shift[2])
    if dz == 0 and dy == 0 and dx == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=mask.dtype)
    z_src = max(0, -dz); z_dst = max(0, dz); z_len = min(mask.shape[0] - z_src, mask.shape[0] - z_dst)
    y_src = max(0, -dy); y_dst = max(0, dy); y_len = min(mask.shape[1] - y_src, mask.shape[1] - y_dst)
    x_src = max(0, -dx); x_dst = max(0, dx); x_len = min(mask.shape[2] - x_src, mask.shape[2] - x_dst)
    if z_len <= 0 or y_len <= 0 or x_len <= 0:
        return out
    out[z_dst:z_dst + z_len, y_dst:y_dst + y_len, x_dst:x_dst + x_len] = \
        mask[z_src:z_src + z_len, y_src:y_src + y_len, x_src:x_src + x_len]
    return out


def _superior_offset_vox(
    voxel_spacing: tuple,
    offset_mm: float = 20.0,
) -> np.ndarray:
    """
    Shift along -Z (superior direction) in voxel space (z,y,x).
    Positive offset_mm moves toward smaller z indices.
    """
    spacing_vec = np.array(voxel_spacing, dtype=float)
    dz = float(spacing_vec[0]) if spacing_vec[0] != 0 else 1.0
    offset_vox = np.zeros(3, dtype=float)
    offset_vox[0] = -float(offset_mm) / dz
    return offset_vox


# ============================================================
#  ANATOMICAL CROSS-SECTION SAMPLING
# ============================================================

def _vaginal_cross_section(vagina_voxels, entry_center, u, v, w, thickness=1.0):
    """
    Extract 2D anatomical cross-section of the vagina.

    vagina_voxels : (N,3) (z,y,x)
    entry_center  : voxel coords (z,y,x)
    u,v,w         : basis vectors (z,y,x)
    thickness     : how thick a slice along w to accept
    """
    pts = vagina_voxels.astype(float)
    rel = pts - entry_center[None, :]

    # Project onto canal direction
    dist_w = rel @ w
    mask = np.abs(dist_w) <= thickness
    slice_pts = pts[mask]

    if slice_pts.shape[0] == 0:
        return None, None

    rel2 = slice_pts - entry_center[None, :]
    U = rel2 @ u
    V = rel2 @ v

    return np.column_stack([U, V]), slice_pts


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.unique(points.astype(float), axis=0)
    if pts.shape[0] <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull



def _point_in_poly(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _sample_points_in_polygon(poly: np.ndarray, count: int, rng, max_attempts: int = 50000):
    if poly.shape[0] < 3 or count <= 0:
        return []
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    samples = []
    attempts = 0
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float)
        if _point_in_poly(p, poly):
            samples.append(p)
    return samples


# ============================================================
#  CANDIDATE NEEDLE HELPERS (MM-TRUE EDT / DILATION)
# ============================================================

def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def _sample_points_from_mask_surface(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Surface voxels (z,y,x) from a binary mask by taking a 1-voxel shell."""
    eroded = ndi.binary_erosion(mask, iterations=1)
    shell = mask & (~eroded)
    idx = np.argwhere(shell)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=int)
    if idx.shape[0] <= n:
        return idx
    sel = rng.choice(idx.shape[0], size=n, replace=False)
    return idx[sel]


def _dilated_surface_points_mm(mask: np.ndarray,
                               n: int,
                               rng: np.random.Generator,
                               dilation_mm: float,
                               spacing_mm: tuple[float, float, float]) -> np.ndarray:
    """Points just outside the mask surface within dilation_mm (mm-true)."""
    dist_outside_mm = ndi.distance_transform_edt(~mask, sampling=spacing_mm)
    band = (dist_outside_mm > 0.0) & (dist_outside_mm <= float(dilation_mm))
    idx = np.argwhere(band)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=int)
    if idx.shape[0] <= n:
        return idx
    sel = rng.choice(idx.shape[0], size=n, replace=False)
    return idx[sel]


def _min_energy_quintic_control_points(P0: np.ndarray,
                                       P1: np.ndarray,
                                       P4: np.ndarray,
                                       P5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form min-bending-energy solution for quintic Bézier interior points."""
    P2 = (-0.3 * P0) + (1.0 * P1) + (0.5 * P4) - (0.2 * P5)
    P3 = (-0.2 * P0) + (0.5 * P1) + (1.0 * P4) - (0.3 * P5)
    return P2, P3


def _sample_quintic_bezier(P0: np.ndarray,
                           P1: np.ndarray,
                           P2: np.ndarray,
                           P3: np.ndarray,
                           P4: np.ndarray,
                           P5: np.ndarray,
                           n_points: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(2, int(n_points)))[:, None]
    B0 = (1 - t) ** 5
    B1 = 5 * (1 - t) ** 4 * t
    B2 = 10 * (1 - t) ** 3 * t ** 2
    B3 = 10 * (1 - t) ** 2 * t ** 3
    B4 = 5 * (1 - t) * t ** 4
    B5 = t ** 5
    return B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3 + B4 * P4 + B5 * P5


def _min_energy_quintic_polyline(entry_zyx: np.ndarray,
                                 target_zyx: np.ndarray,
                                 t0_dir_mm: np.ndarray,
                                 t1_dir_mm: np.ndarray,
                                 spacing_mm: tuple[float, float, float],
                                 t0_len_mm: float,
                                 t1_len_mm: float,
                                 n_points: int) -> np.ndarray:
    spacing = np.asarray(spacing_mm, dtype=float)
    P0 = entry_zyx.astype(float) * spacing
    P5 = target_zyx.astype(float) * spacing

    t0 = t0_dir_mm / (np.linalg.norm(t0_dir_mm) + 1e-8)
    t1 = t1_dir_mm / (np.linalg.norm(t1_dir_mm) + 1e-8)

    # Quintic endpoint tangents: C'(0)=5(P1-P0), C'(1)=5(P5-P4)
    P1 = P0 + t0 * (float(t0_len_mm) / 5.0)
    P4 = P5 - t1 * (float(t1_len_mm) / 5.0)

    P2, P3 = _min_energy_quintic_control_points(P0, P1, P4, P5)
    pts_mm = _sample_quintic_bezier(P0, P1, P2, P3, P4, P5, n_points=n_points)
    pts_zyx = pts_mm / spacing
    return np.round(pts_zyx).astype(int)


def _min_dist_between_polylines_mm(poly_a_zyx: np.ndarray,
                                   poly_b_zyx: np.ndarray,
                                   spacing_mm: tuple[float, float, float]) -> float:
    if poly_a_zyx.size == 0 or poly_b_zyx.size == 0:
        return np.inf
    A = poly_a_zyx.astype(float) * np.array(spacing_mm)[None, :]
    B = poly_b_zyx.astype(float) * np.array(spacing_mm)[None, :]
    d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=-1)
    return float(np.sqrt(d2.min()))


def _build_oar_distance_maps_mm(structure_mask: np.ndarray,
                                oar_labels: list[int],
                                spacing_mm: tuple[float, float, float]) -> dict[int, np.ndarray]:
    """Distance map in mm to nearest OAR voxel (0 inside OAR)."""
    dmaps = {}
    for lbl in oar_labels:
        oar = (structure_mask == lbl)
        dmaps[lbl] = ndi.distance_transform_edt(~oar, sampling=spacing_mm)
    return dmaps




def _passes_oar_clearance_along_polyline_mm(poly_zyx: np.ndarray,
                                            oar_distmaps_mm: dict[int, np.ndarray],
                                            clearance_mm: dict[int, float]) -> bool:
    if poly_zyx is None or poly_zyx.size == 0:
        return False
    if not oar_distmaps_mm:
        return True
    pts = np.asarray(poly_zyx, dtype=int)
    zmax, ymax, xmax = next(iter(oar_distmaps_mm.values())).shape
    pts[:, 0] = np.clip(pts[:, 0], 0, zmax - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, ymax - 1)
    pts[:, 2] = np.clip(pts[:, 2], 0, xmax - 1)

    for lbl, dmap_mm in oar_distmaps_mm.items():
        need_mm = float(clearance_mm.get(lbl, 0.0))
        if need_mm <= 0:
            continue
        d_mm_min = float(np.min(dmap_mm[pts[:, 0], pts[:, 1], pts[:, 2]]))
        if d_mm_min < need_mm:
            return False
    return True


def _passes_angle_to_tandem(entry_zyx: np.ndarray,
                            target_zyx: np.ndarray,
                            tandem_axis_zyx: np.ndarray,
                            spacing_mm: tuple[float, float, float],
                            max_angle_deg: float) -> bool:
    needle_dir_mm = (target_zyx.astype(float) - entry_zyx.astype(float)) * np.array(spacing_mm)
    axis_mm = tandem_axis_zyx.astype(float) * np.array(spacing_mm)
    if np.linalg.norm(needle_dir_mm) < 1e-6 or np.linalg.norm(axis_mm) < 1e-6:
        return False
    return _angle_between(needle_dir_mm, axis_mm) <= float(max_angle_deg)


def _passes_angle_to_tandem_in_plane(entry_zyx: np.ndarray,
                                     target_zyx: np.ndarray,
                                     tandem_axis_zyx: np.ndarray,
                                     spacing_mm: tuple[float, float, float],
                                     plane_u_zyx: np.ndarray,
                                     plane_v_zyx: np.ndarray,
                                     max_angle_deg: float) -> bool:
    spacing_vec = np.array(spacing_mm, dtype=float)
    needle_dir_mm = (target_zyx.astype(float) - entry_zyx.astype(float)) * spacing_vec
    axis_mm = tandem_axis_zyx.astype(float) * spacing_vec
    if np.linalg.norm(needle_dir_mm) < 1e-6 or np.linalg.norm(axis_mm) < 1e-6:
        return False

    u_mm = _unit(np.asarray(plane_u_zyx, dtype=float) * spacing_vec)
    v_mm = _unit(np.asarray(plane_v_zyx, dtype=float) * spacing_vec)
    if np.linalg.norm(u_mm) < 1e-6 or np.linalg.norm(v_mm) < 1e-6:
        return _passes_angle_to_tandem(entry_zyx, target_zyx, tandem_axis_zyx, spacing_mm, max_angle_deg)

    axis_proj = u_mm * np.dot(axis_mm, u_mm) + v_mm * np.dot(axis_mm, v_mm)
    dir_proj = u_mm * np.dot(needle_dir_mm, u_mm) + v_mm * np.dot(needle_dir_mm, v_mm)
    if np.linalg.norm(axis_proj) < 1e-6 or np.linalg.norm(dir_proj) < 1e-6:
        # If projection is degenerate, skip the constraint rather than hard-fail.
        return True
    return _angle_between(axis_proj, dir_proj) <= float(max_angle_deg)


def _biased_target_sampling(entry_zyx: np.ndarray,
                            surface_pts_zyx: np.ndarray,
                            rng: np.random.Generator,
                            k: int,
                            spacing_mm: tuple[float, float, float],
                            cone_axis_zyx: np.ndarray | None = None,
                            cone_half_angle_deg: float = 45.0) -> np.ndarray:
    if surface_pts_zyx.size == 0:
        return np.zeros((0, 3), dtype=int)

    spacing_vec = np.array(spacing_mm, dtype=float)
    if cone_axis_zyx is None:
        centroid = surface_pts_zyx.mean(axis=0)
        cone_axis_mm = _unit((centroid - entry_zyx.astype(float)) * spacing_vec)
    else:
        cone_axis_mm = _unit(cone_axis_zyx.astype(float) * spacing_vec)

    vecs_mm = (surface_pts_zyx.astype(float) - entry_zyx.astype(float)) * spacing_vec
    norms = np.linalg.norm(vecs_mm, axis=1) + 1e-8
    vecs_u = vecs_mm / norms[:, None]
    cosang = np.clip(np.sum(vecs_u * cone_axis_mm[None, :], axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))

    in_cone = surface_pts_zyx[ang <= cone_half_angle_deg]
    pool = in_cone if in_cone.shape[0] >= max(10, k) else surface_pts_zyx

    if pool.shape[0] <= k:
        return pool
    sel = rng.choice(pool.shape[0], size=k, replace=False)
    return pool[sel]


def _hrctv_coverage_score(candidate_line_pts: np.ndarray,
                          hrctv_mask: np.ndarray,
                          cover_radius_vox: int = 1) -> int:
    if candidate_line_pts.size == 0:
        return 0
    zmax, ymax, xmax = hrctv_mask.shape
    tmp = np.zeros_like(hrctv_mask, dtype=bool)
    pts = candidate_line_pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, zmax - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, ymax - 1)
    pts[:, 2] = np.clip(pts[:, 2], 0, xmax - 1)
    tmp[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    tmp = ndi.binary_dilation(tmp, iterations=int(cover_radius_vox))
    return int(np.sum(tmp & hrctv_mask))


def _prune_candidates_greedy(candidates: list[dict],
                             spacing_mm: tuple[float, float, float],
                             max_keep: int = 60,
                             min_sep_mm: float = 3.0) -> list[dict]:
    cands = sorted(candidates, key=lambda d: d["score"], reverse=True)
    kept = []
    for c in cands:
        ok = True
        for k in kept:
            if _min_dist_between_polylines_mm(
                c["polyline_zyx"], k["polyline_zyx"], spacing_mm
            ) < float(min_sep_mm):
                ok = False
                break
        if ok:
            kept.append(c)
        if len(kept) >= max_keep:
            break
    return kept


def _build_candidate_needles_224_lite(structure_mask: np.ndarray,
                                      label_mapping: dict[str, int],
                                      entry_points_zyx: np.ndarray,
                                      tandem_axis_zyx: np.ndarray,
                                      spacing_mm: tuple[float, float, float],
                                      n_targets_per_entry: int = 25,
                                      n_outside_targets_per_entry: int = 10,
                                      outside_dilation_mm: float = 3.0,
                                      max_angle_to_tandem_deg: float = 35.0,
                                      oar_clearance_mm: dict[str, float] | None = None,
                                      rng_seed: int = 7,
                                      prune_max_keep: int = 80,
                                      prune_min_sep_mm: float = 3.0,
                                      score_hrctv: bool = True,
                                      target_center_zyx: np.ndarray | None = None,
                                      target_u_zyx: np.ndarray | None = None,
                                      target_v_zyx: np.ndarray | None = None,
                                      target_hull_uv: np.ndarray | None = None,
                                      target_radius_mm: float | None = None,
                                      debug_stats: dict | None = None) -> list[dict]:
    rng = np.random.default_rng(rng_seed)

    hrctv_label = label_mapping.get("HRCTV", 1)
    hrctv_mask = (structure_mask == hrctv_label)

    if oar_clearance_mm is None:
        oar_clearance_mm = {
            "Rectum": 3.0,
            "Bladder": 3.0,
            "Sigmoid": 3.0,
            "Bowel": 3.0,
            "Vagina": 0.0,
        }
    oar_labels = [label_mapping[name] for name in oar_clearance_mm.keys() if name in label_mapping]
    oar_clearance_by_lbl = {label_mapping[name]: float(mm)
                            for name, mm in oar_clearance_mm.items()
                            if name in label_mapping}
    oar_distmaps = _build_oar_distance_maps_mm(structure_mask, oar_labels, spacing_mm)

    use_target_plane = (
        target_center_zyx is not None
        and target_u_zyx is not None
        and target_v_zyx is not None
    )
    if not use_target_plane:
        hrctv_surface = _sample_points_from_mask_surface(hrctv_mask, n=12000, rng=rng)
        outside_surface = _dilated_surface_points_mm(
            hrctv_mask, n=8000, rng=rng, dilation_mm=outside_dilation_mm, spacing_mm=spacing_mm
        )

    if debug_stats is not None:
        debug_stats.setdefault("entries_total", int(len(entry_points_zyx)))
        debug_stats.setdefault("targets_total", 0)
        debug_stats.setdefault("angle_reject", 0)
        debug_stats.setdefault("zero_len_reject", 0)
        debug_stats.setdefault("oar_reject", 0)
        debug_stats.setdefault("candidates_preprune", 0)

    candidates = []
    spacing_vec = np.array(spacing_mm, dtype=float)
    min_step_mm = float(np.min(spacing_vec)) * 0.5
    t0_axis_mm = tandem_axis_zyx.astype(float) * spacing_vec
    for entry in entry_points_zyx:
        entry = np.array(entry, dtype=int)

        if use_target_plane:
            total_targets = max(1, int(n_targets_per_entry) + int(n_outside_targets_per_entry))
            targets = []
            tgt_center = np.asarray(target_center_zyx, dtype=float)
            tgt_u = np.asarray(target_u_zyx, dtype=float)
            tgt_v = np.asarray(target_v_zyx, dtype=float)
            if target_hull_uv is not None and len(target_hull_uv) >= 3:
                uv_samples = _sample_points_in_polygon(np.asarray(target_hull_uv, dtype=float), total_targets, rng)
                for uv in uv_samples:
                    tgt = tgt_center + tgt_u * uv[0] + tgt_v * uv[1]
                    targets.append(np.round(tgt).astype(int))
            else:
                radius_mm = float(target_radius_mm) if target_radius_mm is not None else 10.0
                radius_vox = radius_mm / max(float(np.mean(spacing_vec)), 1e-6)
                for _ in range(total_targets):
                    r = radius_vox * np.sqrt(rng.uniform(0.0, 1.0))
                    theta = rng.uniform(0.0, 2.0 * np.pi)
                    uv = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=float)
                    tgt = tgt_center + tgt_u * uv[0] + tgt_v * uv[1]
                    targets.append(np.round(tgt).astype(int))
            if not targets:
                continue
            tgt_candidates = np.vstack(targets)
        else:
            tgt_in = _biased_target_sampling(
                entry, hrctv_surface, rng=rng, k=n_targets_per_entry,
                spacing_mm=spacing_mm, cone_axis_zyx=tandem_axis_zyx,
                cone_half_angle_deg=50.0
            )
            tgt_out = _biased_target_sampling(
                entry, outside_surface, rng=rng, k=n_outside_targets_per_entry,
                spacing_mm=spacing_mm, cone_axis_zyx=tandem_axis_zyx,
                cone_half_angle_deg=60.0
            )
            tgt_candidates = np.vstack([tgt_in, tgt_out])

        if debug_stats is not None:
            debug_stats["targets_total"] += int(len(tgt_candidates))

        for tgt in tgt_candidates:
            tgt = np.array(tgt, dtype=int)

            if use_target_plane:
                if not _passes_angle_to_tandem_in_plane(
                    entry,
                    tgt,
                    tandem_axis_zyx,
                    spacing_mm,
                    target_u_zyx,
                    target_v_zyx,
                    max_angle_to_tandem_deg,
                ):
                    if debug_stats is not None:
                        debug_stats["angle_reject"] += 1
                    continue
            elif not _passes_angle_to_tandem(entry, tgt, tandem_axis_zyx, spacing_mm, max_angle_to_tandem_deg):
                if debug_stats is not None:
                    debug_stats["angle_reject"] += 1
                continue

            entry_mm = entry.astype(float) * spacing_vec
            tgt_mm = tgt.astype(float) * spacing_vec
            seg_mm = tgt_mm - entry_mm
            dist_mm = float(np.linalg.norm(seg_mm))
            if dist_mm < 1e-6:
                if debug_stats is not None:
                    debug_stats["zero_len_reject"] += 1
                continue

            t0_len_mm = float(np.clip(0.2 * dist_mm, 5.0, 25.0))
            t1_len_mm = float(np.clip(0.2 * dist_mm, 5.0, 25.0))
            t1_dir_mm = seg_mm

            n_points = max(10, int(np.ceil(dist_mm / (min_step_mm + 1e-8))) + 1)
            n_points = min(200, n_points)

            poly = _min_energy_quintic_polyline(
                entry_zyx=entry,
                target_zyx=tgt,
                t0_dir_mm=t0_axis_mm,
                t1_dir_mm=t1_dir_mm,
                spacing_mm=spacing_mm,
                t0_len_mm=t0_len_mm,
                t1_len_mm=t1_len_mm,
                n_points=n_points,
            )

            if not _passes_oar_clearance_along_polyline_mm(poly, oar_distmaps, oar_clearance_by_lbl):
                if debug_stats is not None:
                    debug_stats["oar_reject"] += 1
                continue

            if score_hrctv:
                score = _hrctv_coverage_score(poly, hrctv_mask, cover_radius_vox=1)
            else:
                score = 0

            candidates.append({
                "entry_zyx": entry,
                "target_zyx": tgt,
                "polyline_zyx": poly,
                "score": score,
            })

    if debug_stats is not None:
        debug_stats["candidates_preprune"] = int(len(candidates))

    kept = _prune_candidates_greedy(
        candidates, spacing_mm,
        max_keep=prune_max_keep,
        min_sep_mm=prune_min_sep_mm
    )
    if debug_stats is not None:
        debug_stats["candidates_postprune"] = int(len(kept))
        # Capture polylines for visualization (pre/post prune).
        debug_stats["candidates_preprune_list"] = [c["polyline_zyx"] for c in candidates]
        debug_stats["candidates_postprune_list"] = [c["polyline_zyx"] for c in kept]
    return kept

# ============================================================
#  EXIT MARCHING + BEZIER CURVE
# ============================================================



def _angle_between(vec_a, vec_b):
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cos = np.clip(np.dot(vec_a, vec_b) / (na * nb), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _resample_curve(curve: np.ndarray, voxel_spacing, step_mm: float) -> np.ndarray:
    if curve.shape[0] <= 1 or step_mm <= 0:
        return curve

    spacing = np.asarray(voxel_spacing, dtype=float)
    resampled = [curve[0].copy()]
    dist_since = 0.0
    prev = curve[0].copy()

    for i in range(1, curve.shape[0]):
        current = curve[i].copy()
        seg_vec = current - prev
        seg_len = np.linalg.norm(seg_vec * spacing)
        if seg_len < 1e-6:
            continue

        while dist_since + seg_len >= step_mm:
            needed = step_mm - dist_since
            frac = needed / seg_len
            new_point = prev + seg_vec * frac
            resampled.append(new_point.copy())
            prev = new_point
            seg_vec = current - prev
            seg_len = np.linalg.norm(seg_vec * spacing)
            dist_since = 0.0
            if seg_len < 1e-6:
                break

        if seg_len >= 1e-6:
            dist_since += seg_len
            prev = current

    if np.linalg.norm((resampled[-1] - curve[-1]) * spacing) > 1e-3:
        resampled.append(curve[-1])

    return np.asarray(resampled)

DEFAULT_VOLUME_PATH = "/Users/gmoney/Desktop/RLResearch/BrachyRL/data/Pt1 Fx1/Volume/merged_labeled_volume.nii.gz"
# ============================================================
#  BUILD STRAIGHT NEEDLE LIBRARY (WITH REAL CANAL ENTRY/EXIT)
# ============================================================

def build_bent_needle_library(
    structure_mask,
    label_mapping,
    voxel_spacing,
    depth_cm=2.0,
    num_needles=10,
    curve_points=60,
    rng_seed=42,
    slice_thickness_vox=1.5,
    min_entry_separation_mm=None,
    dwell_step_mm=5.0,
    min_path_separation_mm=None,
    os_vox=None,
    entry_radius_mm=20.0,
    entry_angle_limit_deg=45.0,
    world_origin=None,
    allow_vagina_path=True,
    return_entry_plane=False,
    tip_exclusion_mm=7.0,
    score_hrctv=True,
    oar_clearance_mm=None,
    axis_hint_zyx=None,
    debug_rejections: bool = False,
):
    """
    Returns:
    --------
    anatomical_library : list of dicts
        Each dict contains:
           "path_vox" : list[(z,y,x)]
    If return_entry_plane is True, returns (anatomical_library, entry_plane_dict).
    tip_exclusion_mm excludes dwell positions within this distance from the tip.
    """

    HRCTV_LABEL  = label_mapping["HRCTV"]
    VAGINA_LABEL = label_mapping["Vagina"]

    hrctv_mask   = (structure_mask == HRCTV_LABEL)
    vagina_mask  = (structure_mask == VAGINA_LABEL)

    hrctv_voxels  = np.argwhere(hrctv_mask)
    vagina_voxels = np.argwhere(vagina_mask)

    if hrctv_voxels.size == 0:
        raise ValueError("HRCTV mask empty.")
    if vagina_voxels.size == 0:
        raise ValueError("Vagina mask empty.")

   
    spacing_vec = np.array(voxel_spacing, dtype=float)  # (dz,dy,dx)

    # PCA axis in physical (mm) space for fallback alignment with the lumen.
    vagina_mm = vagina_voxels.astype(float) * spacing_vec
    vagina_axis_mm = _pca_major_axis(vagina_mm)
    if np.linalg.norm(vagina_axis_mm) < 1e-8:
        vagina_axis_mm = np.array([0.0, 0.0, 1.0], dtype=float)
    vagina_axis_mm = vagina_axis_mm / (np.linalg.norm(vagina_axis_mm) + 1e-8)

    # Prefer tandem axis for entry plane normal when provided.
    axis_dir_mm = vagina_axis_mm
    if axis_hint_zyx is not None:
        hint = np.asarray(axis_hint_zyx, dtype=float)
        if hint.ndim == 2 and hint.shape[1] == 3 and hint.shape[0] >= 2:
            hint = hint[-1] - hint[0]
        if hint.shape == (3,):
            hint_mm = hint * spacing_vec
            if np.linalg.norm(hint_mm) > 1e-6:
                axis_dir_mm = hint_mm / (np.linalg.norm(hint_mm) + 1e-8)

    # Identify inferior/superior ends using projections along chosen axis.
    proj_mm = vagina_mm @ axis_dir_mm
    min_proj = float(np.min(proj_mm))
    max_proj = float(np.max(proj_mm))
    eps_mm = max(1.0, float(slice_thickness_vox)) * float(np.mean(spacing_vec))

    min_mask = proj_mm <= (min_proj + eps_mm)
    max_mask = proj_mm >= (max_proj - eps_mm)
    min_vox = vagina_voxels[min_mask]
    max_vox = vagina_voxels[max_mask]

    if min_vox.size == 0 or max_vox.size == 0:
        top_idx = int(np.argmin(proj_mm))
        bot_idx = int(np.argmax(proj_mm))
        min_center = vagina_voxels[top_idx].astype(float)
        max_center = vagina_voxels[bot_idx].astype(float)
    else:
        min_center = min_vox.mean(axis=0)
        max_center = max_vox.mean(axis=0)

    # Choose inferior end as larger z index (z+ is inferior in this voxel convention).
    if min_center[0] >= max_center[0]:
        entry_origin = min_center
        superior_center = max_center
    else:
        entry_origin = max_center
        superior_center = min_center

    entry_origin_mm = entry_origin * spacing_vec
    superior_mm = superior_center * spacing_vec
    if axis_hint_zyx is not None:
        canal_axis_mm = axis_dir_mm.copy()
        if float(np.dot(canal_axis_mm, superior_mm - entry_origin_mm)) < 0.0:
            canal_axis_mm = -canal_axis_mm
    else:
        canal_axis_mm = superior_mm - entry_origin_mm
        if np.linalg.norm(canal_axis_mm) < 1e-6:
            canal_axis_mm = vagina_axis_mm
        canal_axis_mm = canal_axis_mm / (np.linalg.norm(canal_axis_mm) + 1e-8)

    posterior_shift_mm = 20.0
    hrctv_centroid = hrctv_voxels.mean(axis=0)
    target_center_mm = hrctv_centroid * spacing_vec
    # Posterior shift along negative z-axis, anchored at posterior-most HRCTV boundary.
    posterior_edge_z = float(np.min(hrctv_voxels[:, 0]))
    target_center_mm[0] = posterior_edge_z * spacing_vec[0] - posterior_shift_mm
    target_center = target_center_mm / spacing_vec
    target_center = np.clip(
        target_center,
        [0, 0, 0],
        np.array(structure_mask.shape) - 1,
    )

    # Convert axis back to voxel space for downstream indexing.
    canal_axis = canal_axis_mm / spacing_vec
    # Entry plane normal: perpendicular to x-axis (plane contains x-axis).
    plane_normal = canal_axis.copy()
    plane_normal[2] = 0.0
    if np.linalg.norm(plane_normal) < 1e-6:
        plane_normal = np.array([1.0, 0.0, 0.0], dtype=float)
    plane_u, plane_v, plane_w = _make_orthonormal_basis(plane_normal)

    # Entry plane center = depth_cm inside canal from the inferior end.
    depth_mm = depth_cm * 10.0
    proj_mm_from_entry = (vagina_mm - entry_origin_mm[None, :]) @ canal_axis_mm
    max_depth_mm = float(np.max(proj_mm_from_entry))
    depth_mm = min(depth_mm, max_depth_mm)

    slab_mm = max(3.0, float(slice_thickness_vox)) * float(np.mean(spacing_vec))
    slice_mask = np.abs(proj_mm_from_entry - depth_mm) <= slab_mm
    if np.any(slice_mask):
        entry_center = vagina_voxels[slice_mask].mean(axis=0)
    else:
        # Fallback: move along axis from inferior end.
        def _offset_along_vec(direction_vec, distance_mm):
            world_vec = direction_vec * spacing_vec
            norm = np.linalg.norm(world_vec)
            if norm < 1e-8:
                return np.zeros(3, dtype=float)
            return direction_vec * (distance_mm / norm)
        entry_center = entry_origin + _offset_along_vec(canal_axis, depth_mm)
    plane_origin = entry_center.copy()

    # Extract anatomical cross-section
    slab_thickness = max(3.0, float(slice_thickness_vox) * 2.0)
    UV, slice_pts = _vaginal_cross_section(
        vagina_voxels,
        plane_origin,
        plane_u,
        plane_v,
        plane_w,
        thickness=slab_thickness,
    )
    if UV is None:
        raise RuntimeError("Could not obtain vaginal cross-section.")

    rng = np.random.default_rng(rng_seed)
    hull = _convex_hull_2d(UV) if UV is not None else None
    lumen_center_uv = None
    if hull is not None and hull.shape[0] >= 3:
        lumen_samples = max(200, num_needles * 20)
        uv_samples = _sample_points_in_polygon(hull, lumen_samples, rng)
        if uv_samples:
            lumen_uv = []
            shape_max = np.array(structure_mask.shape) - 1
            for uv in uv_samples:
                entry = plane_origin + plane_u * uv[0] + plane_v * uv[1]
                entry_vox = np.clip(np.round(entry).astype(int), [0, 0, 0], shape_max)
                if structure_mask[tuple(entry_vox)] == 0:
                    lumen_uv.append(uv)
            if lumen_uv:
                lumen_center_uv = np.mean(np.asarray(lumen_uv, dtype=float), axis=0)
                entry_center = plane_origin + plane_u * lumen_center_uv[0] + plane_v * lumen_center_uv[1]
                hull = hull - lumen_center_uv
            else:
                print("[WARN] No zero-labeled lumen voxels found at entry plane; using plane origin.")
    if hull is not None and hull.shape[0] >= 3:
        hull_pts = entry_center[None, :] + plane_u[None, :] * hull[:, :1] + plane_v[None, :] * hull[:, 1:2]
        entry_radius_mm = float(
            np.max(np.linalg.norm((hull_pts - entry_center[None, :]) * spacing_vec, axis=1))
        )
    else:
        hull = None
        entry_radius_mm = None

    def _dist_mm(p, q):
        diff = (np.asarray(p) - np.asarray(q)) * spacing_vec
        return np.sqrt((diff**2).sum())

    def _norm_mm(vec):
        return np.linalg.norm(vec * spacing_vec)

    if world_origin is None:
        world_origin = (0.0, 0.0, 0.0)

    min_sep_mm = 0.0 if min_entry_separation_mm is None else max(min_entry_separation_mm, 0.0)

    def _add_entry(entry):
        if entry_radius_mm is not None and _norm_mm(entry - entry_center) > entry_radius_mm:
            return False
        entry_vox = np.clip(
            np.round(entry).astype(int),
            [0, 0, 0],
            np.array(structure_mask.shape) - 1,
        )
        lbl = structure_mask[tuple(entry_vox)]
        if lbl != 0:   # inside lumen/background
            return False
        if min_sep_mm > 0:
            for existing in entry_points:
                if _dist_mm(existing, entry) < min_sep_mm:
                    return False
        entry_points.append(entry)
        return True

    entry_points = []
    candidate_entries = []
    rejected_entries = []
    target_entries = max(num_needles, num_needles * 2)
    if hull is not None and len(hull) >= 3:
        inner_uv = _sample_points_in_polygon(hull, target_entries, rng)
        for uv in inner_uv:
            entry = entry_center + plane_u * uv[0] + plane_v * uv[1]
            candidate_entries.append(entry)
            if not _add_entry(entry):
                rejected_entries.append(entry)
    if not entry_points:
        entry_points = [entry_origin.copy()]

    if len(entry_points) < num_needles and min_sep_mm > 0:
        print(
            f"[WARN] Only sampled {len(entry_points)} entry points (target {num_needles}) given min separation {min_sep_mm} mm"
        )

    # Build trajectories (candidate generator only)
    anatomical_library = []
    spacing_vec = np.array(voxel_spacing, dtype=float)

    forbidden_labels = {
        label_mapping.get("Rectum"),
        label_mapping.get("Bladder"),
        label_mapping.get("Sigmoid"),
        label_mapping.get("Bowel"),
        label_mapping.get("Ureter"),
        label_mapping.get("Ureters"),
        label_mapping.get("ParametrialVessels"),
        label_mapping.get("Vessels"),
    }
    tip_forbidden_labels = {
        label_mapping.get("ParametrialVessels"),
        label_mapping.get("Vessels"),
    }
    tip_forbidden_labels = {label for label in tip_forbidden_labels if label is not None}
    allowed_labels = {0, label_mapping.get("HRCTV", -1)}
    if allow_vagina_path:
        allowed_labels.add(label_mapping.get("Vagina", -1))

    entry_points_arr = np.asarray(entry_points, dtype=float)
    if entry_points_arr.size == 0:
        entry_points_arr = np.asarray([entry_origin.copy()], dtype=float)

    prune_min_sep = float(min_path_separation_mm) if min_path_separation_mm is not None else 3.0
    prune_max_keep = max(num_needles * 4, num_needles)

    debug_stats = {} if debug_rejections else None
    target_radius_mm = None if entry_radius_mm is None else float(entry_radius_mm) * 1.3
    candidates = _build_candidate_needles_224_lite(
        structure_mask=structure_mask,
        label_mapping=label_mapping,
        entry_points_zyx=entry_points_arr,
        tandem_axis_zyx=canal_axis,
        spacing_mm=voxel_spacing,
        n_targets_per_entry=25,
        n_outside_targets_per_entry=10,
        outside_dilation_mm=3.0,
        max_angle_to_tandem_deg=entry_angle_limit_deg,
        oar_clearance_mm=oar_clearance_mm,
        rng_seed=rng_seed,
        prune_max_keep=prune_max_keep,
        prune_min_sep_mm=prune_min_sep,
        score_hrctv=score_hrctv,
        target_center_zyx=target_center,
        target_u_zyx=plane_u,
        target_v_zyx=plane_v,
        target_hull_uv=hull,
        target_radius_mm=target_radius_mm,
        debug_stats=debug_stats,
    )
    if debug_stats is not None:
        debug_stats["candidates_postprune"] = int(len(candidates))
        debug_stats["entry_points"] = int(len(entry_points))
        debug_stats["candidate_entries"] = int(len(candidate_entries))
        debug_stats["rejected_entries"] = int(len(rejected_entries))

    reject_short_curve = 0
    reject_blocked_label = 0
    reject_not_allowed = 0
    reject_no_hrctv = 0
    accept_near_hrctv = 0
    use_target_plane = True
    hrctv_distmap_mm = None
    hrctv_proximity_mm = 5.0
    reject_tip_forbidden = 0
    reject_os_proj = 0
    reject_tip_exclusion = 0

    for cand in candidates:
        if len(anatomical_library) >= num_needles:
            break
        curve = np.asarray(cand["polyline_zyx"], dtype=float)
        if curve.shape[0] < 2:
            reject_short_curve += 1
            continue
        curve = _resample_curve(curve, voxel_spacing, dwell_step_mm)
        curve[:, 0] = np.clip(curve[:, 0], 0, structure_mask.shape[0] - 1)
        curve[:, 1] = np.clip(curve[:, 1], 0, structure_mask.shape[1] - 1)
        curve[:, 2] = np.clip(curve[:, 2], 0, structure_mask.shape[2] - 1)

        path = []
        last = None
        blocked = False
        for p in curve.astype(int):
            tup = (p[0], p[1], p[2])
            if tup != last:
                label = structure_mask[tup[0], tup[1], tup[2]]
                if label in forbidden_labels and label is not None:
                    reject_blocked_label += 1
                    blocked = True
                    break
                if label not in allowed_labels:
                    reject_not_allowed += 1
                    blocked = True
                    break
                path.append(tup)
                last = tup

        if blocked or len(path) == 0:
            continue

        intersects_hrctv = any(structure_mask[z, y, x] == HRCTV_LABEL for z, y, x in path)
        if not intersects_hrctv:
            if use_target_plane:
                if hrctv_distmap_mm is None:
                    hrctv_distmap_mm = ndi.distance_transform_edt(~hrctv_mask, sampling=spacing_vec)
                path_arr = np.asarray(path, dtype=int)
                zmax, ymax, xmax = hrctv_distmap_mm.shape
                path_arr[:, 0] = np.clip(path_arr[:, 0], 0, zmax - 1)
                path_arr[:, 1] = np.clip(path_arr[:, 1], 0, ymax - 1)
                path_arr[:, 2] = np.clip(path_arr[:, 2], 0, xmax - 1)
                min_hrctv_dist = float(np.min(hrctv_distmap_mm[path_arr[:, 0], path_arr[:, 1], path_arr[:, 2]]))
                if min_hrctv_dist <= hrctv_proximity_mm:
                    accept_near_hrctv += 1
                else:
                    reject_no_hrctv += 1
                    continue
            else:
                reject_no_hrctv += 1
                continue
        tip_label = structure_mask[path[-1][0], path[-1][1], path[-1][2]]
        if tip_label in tip_forbidden_labels and tip_label is not None:
            reject_tip_forbidden += 1
            continue

        min_dwell_idx = 0
        if os_vox is not None:
            os_point = np.asarray(os_vox, dtype=float)
            axis_dir_mm = canal_axis_mm / (np.linalg.norm(canal_axis_mm) + 1e-8)
            proj = ((np.asarray(path, dtype=float) - os_point) * spacing_vec) @ axis_dir_mm
            valid_mask = proj >= 0.0
            if not np.any(valid_mask):
                reject_os_proj += 1
                continue
            min_dwell_idx = int(np.argmax(valid_mask))

        max_dwell_idx = len(path) - 1
        if tip_exclusion_mm is not None and tip_exclusion_mm > 0.0:
            path_arr = np.asarray(path, dtype=float)
            if path_arr.shape[0] < 2:
                continue
            diffs = np.diff(path_arr, axis=0) * spacing_vec
            seg_mm = np.linalg.norm(diffs, axis=1)
            dist_from_entry = np.concatenate([[0.0], np.cumsum(seg_mm)])
            total_len = dist_from_entry[-1]
            if total_len <= tip_exclusion_mm:
                reject_tip_exclusion += 1
                continue
            dist_from_tip = total_len - dist_from_entry
            valid_mask = dist_from_tip >= (tip_exclusion_mm - 1e-6)
            if not np.any(valid_mask):
                reject_tip_exclusion += 1
                continue
            max_dwell_idx = int(np.max(np.where(valid_mask)[0]))
        if max_dwell_idx < min_dwell_idx:
            reject_tip_exclusion += 1
            continue

        anatomical_library.append({
            "path_vox": path,
            "min_dwell_idx": min_dwell_idx,
            "max_dwell_idx": max_dwell_idx,
        })

    print(f"[INFO] Built {len(anatomical_library)} anatomically-sampled needles.")
    if debug_rejections:
        print("[DEBUG] Needle library rejection summary:")
        if debug_stats is not None:
            print(
                "  entries_total={entries_total} entry_points={entry_points} "
                "candidate_entries={candidate_entries} rejected_entries={rejected_entries}".format(**debug_stats)
            )
            print(
                "  targets_total={targets_total} angle_reject={angle_reject} "
                "zero_len_reject={zero_len_reject} oar_reject={oar_reject}".format(**debug_stats)
            )
            print(
                "  candidates_preprune={candidates_preprune} candidates_postprune={candidates_postprune}".format(**debug_stats)
            )
        print(
            f"  reject_short_curve={reject_short_curve} reject_blocked_label={reject_blocked_label} "
            f"reject_not_allowed={reject_not_allowed} reject_no_hrctv={reject_no_hrctv} "
            f"accept_near_hrctv={accept_near_hrctv}"
        )
        print(
            f"  reject_tip_forbidden={reject_tip_forbidden} reject_os_proj={reject_os_proj} "
            f"reject_tip_exclusion={reject_tip_exclusion}"
        )
    entry_plane = {
        "center_vox": entry_center.astype(float),
        "u": plane_u.astype(float),
        "v": plane_v.astype(float),
        "w": plane_w.astype(float),
        "hull_uv": None if hull is None else np.asarray(hull, dtype=float),
        "entry_radius_mm": None if entry_radius_mm is None else float(entry_radius_mm),
        "entry_origin_vox": entry_origin.astype(float),
        "target_center_vox": target_center.astype(float),
        "target_shift_mm": float(-posterior_shift_mm),
        "candidate_entries_vox": None if not candidate_entries else np.asarray(candidate_entries, dtype=float),
        "rejected_entries_vox": None if not rejected_entries else np.asarray(rejected_entries, dtype=float),
        "accepted_entries_vox": None if not entry_points else np.asarray(entry_points, dtype=float),
    }
    if debug_stats is not None:
        entry_plane["candidate_polylines_preprune"] = debug_stats.get("candidates_preprune_list")
        entry_plane["candidate_polylines_postprune"] = debug_stats.get("candidates_postprune_list")

    if return_entry_plane:
        return anatomical_library, entry_plane
    return anatomical_library


# ============================================================
#  VOXEL → WORLD USING FULL AFFINE
# ============================================================

def vox_to_world(path_vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    path_vox = np.asarray(path_vox, float)
    xyz = path_vox[:, ::-1]                 # (x,y,z)
    homo = np.hstack([xyz, np.ones((len(xyz),1))])
    world = (affine @ homo.T).T
    return world[:, :3]


# ============================================================
#  MASK → PYVISTA MESH USING AFFINE
# ============================================================

def _mask_to_pv_mesh(mask: np.ndarray, affine: np.ndarray):
    if np.sum(mask) == 0:
        return None

    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), 0.5)
    world = vox_to_world(verts, affine)

    faces_pv = np.hstack([np.full((faces.shape[0],1),3,dtype=np.int32),
                          faces.astype(np.int32)]).ravel()
    return pv.PolyData(world, faces_pv)


# ============================================================
#  SIMPLE POLYLINE HELPER
# ============================================================

def _polyline(points: np.ndarray) -> "pv.PolyData":
    pts = np.asarray(points, dtype=float)
    poly = pv.PolyData(pts)
    if pts.shape[0] >= 2:
        poly.lines = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])]).astype(np.int64)
    return poly




# ============================================================
#  VISUALIZATION
# ============================================================

def visualize_bent_needles(
    structure_mask,
    label_mapping,
    anatomical_library,
    affine,
    voxel_spacing=None,
    tandem_paths=None,
    tandem_color="blue",
    ovoid_paths=None,
    ovoid_color="cyan",
    entry_plane=None,
    entry_plane_color="lightgreen",
    entry_plane_opacity=0.3,
    show_entry_debug=False,
    entry_point_size=6,
    entry_candidate_color="silver",
    entry_reject_color="orange",
    entry_accept_color="lime",
    entry_center_color="green",
    entry_origin_color="purple",
    pca_axis=None,
    pca_axis_color="black",
    pca_axis_scale_mm=80.0,
    show_target_zone=True,
    target_color="green",
    target_opacity=0.25,
    screenshot_path=None,
    off_screen=False,
    save_html=None,
    save_vtk=None,
):
    organ_colors = {
        "HRCTV": "red",
        "Rectum": "indigo",
        "Bladder": "deepskyblue",
        "Sigmoid": "orange",
        "Bowel": "purple",
        "Vagina": "pink",
    }

    if voxel_spacing is None:
        col0 = np.linalg.norm(affine[:3,0])
        col1 = np.linalg.norm(affine[:3,1])
        col2 = np.linalg.norm(affine[:3,2])
        voxel_spacing = (col2, col1, col0)

    dz, dy, dx = voxel_spacing
    tube_radius = min(dx, dy, dz) * 0.7

    pv.set_jupyter_backend(None)
    pl = pv.Plotter(window_size=(1200,1000), off_screen=off_screen or screenshot_path is not None)
    pl.set_background("white")

    legend_items = []
    for organ, color in organ_colors.items():
        label = label_mapping.get(organ)
        if label is None: continue
        mask = (structure_mask == label)
        mesh = _mask_to_pv_mesh(mask, affine)
        if mesh is None: continue
        mesh = mesh.smooth(n_iter=20, relaxation_factor=0.1)
        pl.add_mesh(mesh,
                    color=color,
                    opacity=0.2 if organ!="HRCTV" else 0.35,
                    smooth_shading=True)
        legend_items.append((organ, color))

    if show_target_zone and entry_plane is not None:
        target_center_vox = entry_plane.get("target_center_vox")
        if target_center_vox is not None:
            target_center_vox = np.asarray(target_center_vox, dtype=float)
            u = np.asarray(entry_plane.get("u"), dtype=float)
            v = np.asarray(entry_plane.get("v"), dtype=float)
            w = np.asarray(entry_plane.get("w"), dtype=float)
            hull_uv = entry_plane.get("hull_uv")
            radius_mm = entry_plane.get("entry_radius_mm")

            target_mesh = None
            if hull_uv is not None and len(hull_uv) >= 3:
                hull_uv = np.asarray(hull_uv, dtype=float)
                pts_vox = target_center_vox[None, :] + u[None, :] * hull_uv[:, :1] + v[None, :] * hull_uv[:, 1:2]
                pts_world = vox_to_world(pts_vox, affine)
                target_mesh = pv.PolyData(pts_world)
                faces = np.hstack([[pts_world.shape[0]], np.arange(pts_world.shape[0])]).astype(np.int64)
                target_mesh.faces = faces
            elif radius_mm is not None:
                center_world = vox_to_world(target_center_vox[None, :], affine)[0]
                normal_world = vox_to_world((target_center_vox + w)[None, :], affine)[0] - center_world
                if np.linalg.norm(normal_world) < 1e-8:
                    normal_world = np.array([0.0, 0.0, 1.0], dtype=float)
                target_mesh = pv.Disc(
                    center=center_world,
                    normal=normal_world,
                    inner=0.0,
                    outer=float(radius_mm),
                )

            if target_mesh is not None:
                pl.add_mesh(
                    target_mesh,
                    color=target_color,
                    opacity=target_opacity,
                    smooth_shading=True,
                )
                legend_items.append(("Target plane", target_color))

    print(f"[INFO] Visualizing {len(anatomical_library)} needles")

    for i, nd in enumerate(anatomical_library):
        vox = np.array(nd["path_vox"], float)
        pts = vox_to_world(vox, affine)

        spline = pv.Spline(pts, 200)
        tube = spline.tube(radius=tube_radius)
        pl.add_mesh(tube, color="grey", smooth_shading=True)
        pl.add_points(pts, color="lime", point_size=8)
    if anatomical_library:
        legend_items.append(("Needle paths", "black"))
        legend_items.append(("Dwell points", "lime"))

    if tandem_paths:
        print(f"[INFO] Overlaying {len(tandem_paths)} tandem path(s)")
        for path in tandem_paths:
            vox = np.array(path, float)
            if vox.size == 0:
                continue
            pts = vox_to_world(vox, affine)
            spline = pv.Spline(pts, len(pts))
            tube = spline.tube(radius=tube_radius * 1.2)
            pl.add_mesh(tube, color=tandem_color, smooth_shading=True, opacity=0.8)
            pl.add_points(pts, color="cyan", point_size=10)
        legend_items.append(("Tandem", tandem_color))

    if ovoid_paths:
        print(f"[INFO] Overlaying {len(ovoid_paths)} ovoid path(s)")
        for entry in ovoid_paths:
            path = entry.get("path_vox") if isinstance(entry, dict) else entry
            if path is None:
                continue
            vox = np.array(path, float)
            if vox.size == 0:
                continue
            pts = vox_to_world(vox, affine)
            spline = pv.Spline(pts, len(pts))
            tube = spline.tube(radius=tube_radius)
            pl.add_mesh(tube, color=ovoid_color, smooth_shading=True, opacity=0.8)
            pl.add_points(pts, color=ovoid_color, point_size=8)
        legend_items.append(("Ovoids", ovoid_color))

    if entry_plane is not None:
        center_vox = np.asarray(entry_plane.get("center_vox"), dtype=float)
        u = np.asarray(entry_plane.get("u"), dtype=float)
        v = np.asarray(entry_plane.get("v"), dtype=float)
        w = np.asarray(entry_plane.get("w"), dtype=float)
        hull_uv = entry_plane.get("hull_uv")
        radius_mm = entry_plane.get("entry_radius_mm")

        plane_mesh = None
        if hull_uv is not None and len(hull_uv) >= 3:
            hull_uv = np.asarray(hull_uv, dtype=float)
            pts_vox = center_vox[None, :] + u[None, :] * hull_uv[:, :1] + v[None, :] * hull_uv[:, 1:2]
            pts_world = vox_to_world(pts_vox, affine)
            plane_mesh = pv.PolyData(pts_world)
            faces = np.hstack([[pts_world.shape[0]], np.arange(pts_world.shape[0])]).astype(np.int64)
            plane_mesh.faces = faces
        elif radius_mm is not None:
            center_world = vox_to_world(center_vox[None, :], affine)[0]
            normal_world = vox_to_world((center_vox + w)[None, :], affine)[0] - center_world
            if np.linalg.norm(normal_world) < 1e-8:
                normal_world = np.array([0.0, 0.0, 1.0], dtype=float)
            plane_mesh = pv.Disc(
                center=center_world,
                normal=normal_world,
                inner=0.0,
                outer=float(radius_mm),
            )

        if plane_mesh is not None:
            pl.add_mesh(plane_mesh, color=entry_plane_color, opacity=entry_plane_opacity, smooth_shading=True)
            legend_items.append(("Entry plane", entry_plane_color))

        if show_entry_debug:
            center_world = vox_to_world(center_vox[None, :], affine)
            pl.add_points(center_world, color=entry_center_color, point_size=entry_point_size + 4)
            legend_items.append(("Entry center", entry_center_color))

            origin_vox = entry_plane.get("entry_origin_vox")
            if origin_vox is not None:
                origin_world = vox_to_world(np.asarray(origin_vox, dtype=float)[None, :], affine)
                pl.add_points(origin_world, color=entry_origin_color, point_size=entry_point_size + 2)
                legend_items.append(("Entry origin", entry_origin_color))

            cand = entry_plane.get("candidate_entries_vox")
            if cand is not None and len(cand):
                cand_world = vox_to_world(np.asarray(cand, dtype=float), affine)
                pl.add_points(cand_world, color=entry_candidate_color, point_size=entry_point_size)
                legend_items.append(("Entry samples", entry_candidate_color))

            rej = entry_plane.get("rejected_entries_vox")
            if rej is not None and len(rej):
                rej_world = vox_to_world(np.asarray(rej, dtype=float), affine)
                pl.add_points(rej_world, color=entry_reject_color, point_size=entry_point_size)
                legend_items.append(("Entry rejected", entry_reject_color))

            acc = entry_plane.get("accepted_entries_vox")
            if acc is not None and len(acc):
                acc_world = vox_to_world(np.asarray(acc, dtype=float), affine)
                pl.add_points(acc_world, color=entry_accept_color, point_size=entry_point_size + 2)
                legend_items.append(("Entry accepted", entry_accept_color))

    if pca_axis is not None:
        axis_center_vox = None
        if entry_plane is not None:
            axis_center_vox = np.asarray(entry_plane.get("entry_origin_vox"), dtype=float)
        if axis_center_vox is None:
            axis_center_vox = np.zeros(3, dtype=float)
        axis_dir = np.asarray(pca_axis, dtype=float)
        if np.linalg.norm(axis_dir) > 1e-8:
            axis_dir = axis_dir / np.linalg.norm(axis_dir)
            spacing_vec = np.array(voxel_spacing, dtype=float)
            axis_dir_vox = axis_dir / (spacing_vec + 1e-8)
            half = 0.5 * float(pca_axis_scale_mm)
            a = axis_center_vox - axis_dir_vox * half
            b = axis_center_vox + axis_dir_vox * half
            pts_world = vox_to_world(np.vstack([a, b]), affine)
            axis_poly = _polyline(pts_world)
            pl.add_mesh(axis_poly, color=pca_axis_color, line_width=4)
            legend_items.append(("PCA axis", pca_axis_color))

    pl.show_bounds(grid="front", location="outer")
    pl.add_axes(color="black")
    pl.enable_anti_aliasing()
    if legend_items:
        pl.add_legend(legend_items, bcolor="white", border=True)
    if save_vtk:
        os.makedirs(save_vtk, exist_ok=True)
        for organ, color in organ_colors.items():
            label = label_mapping.get(organ)
            if label is None:
                continue
            mask = (structure_mask == label)
            mesh = _mask_to_pv_mesh(mask, affine)
            if mesh is None:
                continue
            mesh.save(os.path.join(save_vtk, f"{organ.lower()}.vtp"))
        needle_poly = None
        for nd in anatomical_library:
            vox = np.array(nd["path_vox"], float)
            if vox.size == 0:
                continue
            pts = vox_to_world(vox, affine)
            poly = pv.Spline(pts, len(pts))
            needle_poly = poly if needle_poly is None else needle_poly.merge(poly)
        if needle_poly is not None:
            needle_poly.save(os.path.join(save_vtk, "needles.vtp"))
        if tandem_paths:
            tandem_poly = None
            for path in tandem_paths:
                vox = np.array(path, float)
                if vox.size == 0:
                    continue
                pts = vox_to_world(vox, affine)
                poly = pv.Spline(pts, len(pts))
                tandem_poly = poly if tandem_poly is None else tandem_poly.merge(poly)
            if tandem_poly is not None:
                tandem_poly.save(os.path.join(save_vtk, "tandem.vtp"))
        if ovoid_paths:
            ovoid_poly = None
            for entry in ovoid_paths:
                path = entry.get("path_vox") if isinstance(entry, dict) else entry
                if path is None:
                    continue
                vox = np.array(path, float)
                if vox.size == 0:
                    continue
                pts = vox_to_world(vox, affine)
                poly = pv.Spline(pts, len(pts))
                ovoid_poly = poly if ovoid_poly is None else ovoid_poly.merge(poly)
            if ovoid_poly is not None:
                ovoid_poly.save(os.path.join(save_vtk, "ovoids.vtp"))
    if save_html:
        out_dir = os.path.dirname(os.path.abspath(save_html))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            pl.export_html(save_html)
        except Exception as exc:
            print(f"[WARN] HTML export failed: {exc}")
    if screenshot_path:
        out_dir = os.path.dirname(os.path.abspath(screenshot_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pl.show(screenshot=screenshot_path, auto_close=True)
    else:
        pl.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build + visualize straight-needle library.")
    parser.add_argument(
        "--volume-path",
        default=DEFAULT_VOLUME_PATH,
        help="Path to merged labeled volume (.nii.gz).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional structure cache dir (overrides --volume-path).",
    )
    parser.add_argument(
        "--screenshot",
        default=None,
        help="Optional screenshot output path for the visualization.",
    )
    parser.add_argument(
        "--min-entry-sep-mm",
        type=float,
        default=None,
        help="Minimum separation (mm) between vaginal entry points.",
    )
    parser.add_argument(
        "--save-html",
        default=None,
        help="Optional HTML export path (requires pyvista[jupyter]).",
    )
    parser.add_argument(
        "--save-vtk",
        default=None,
        help="Optional directory to save VTK meshes.",
    )
    parser.add_argument(
        "--debug-entry-plane",
        action="store_true",
        help="Overlay entry plane sampling (center, hull, sampled points).",
    )
    parser.add_argument(
        "--debug-rejections",
        action="store_true",
        help="Print rejection stats for candidate needle generation.",
    )
    parser.add_argument(
        "--show-pca-axis",
        action="store_true",
        help="Draw the vagina PCA axis used to define the canal direction.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip HRCTV coverage scoring for faster visualization.",
    )
    parser.add_argument(
        "--skip-oar-clearance",
        action="store_true",
        help="Skip OAR distance maps/clearance checks for faster visualization.",
    )
    args = parser.parse_args()

    if args.cache_dir:
        structure_mask, label_mapping, ct_spacing, ct_origin = load_structure_cache(args.cache_dir)
        affine = np.eye(4, dtype=float)
        affine[0, 0] = ct_spacing[0]
        affine[1, 1] = ct_spacing[1]
        affine[2, 2] = ct_spacing[2]
        affine[0, 3] = ct_origin[0]
        affine[1, 3] = ct_origin[1]
        affine[2, 3] = ct_origin[2]
        voxel_spacing = (ct_spacing[2], ct_spacing[1], ct_spacing[0])
    else:
        nii = nib.load(args.volume_path)
        structure_mask = nii.get_fdata().astype(np.int32)
        affine = nii.affine
        voxel_spacing = (
            np.linalg.norm(affine[:3, 2]),  # dz
            np.linalg.norm(affine[:3, 1]),  # dy
            np.linalg.norm(affine[:3, 0]),  # dx
        )
        label_mapping = {
            "HRCTV": 1,
            "Rectum": 2,
            "Bladder": 3,
            "Sigmoid": 4,
            "Bowel": 5,
            "Vagina": 6,
        }

    rtplan_tandem = None
    ovoid_paths = None
    if args.cache_dir:
        meta_path = os.path.join(args.cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise SystemExit("meta.json not found in cache dir; RTPLAN applicators required.")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ct_series = meta.get("ct_series")
        rtstruct = meta.get("rtstruct")
        rtplan = meta.get("rtplan") or _find_rtplan(rtstruct)
        if not ct_series or not rtplan:
            raise SystemExit("RTPLAN applicators required but could not locate ct_series or RTPLAN.")
        rtplan_tandem, ovoid_paths = _load_rtplan_applicators(
            rtplan_path=rtplan,
            ct_series=ct_series,
            structure_mask=structure_mask,
        )
    if rtplan_tandem is None or rtplan_tandem.size == 0:
        raise SystemExit("RTPLAN tandem path not found; cannot proceed.")
    tandem_paths = [rtplan_tandem]

    oar_clearance_mm = {} if args.skip_oar_clearance else None
    anatomical_library, entry_plane = build_bent_needle_library(
        structure_mask=structure_mask,
        label_mapping=label_mapping,
        voxel_spacing=voxel_spacing,
        depth_cm=2.0,
        num_needles=20,
        curve_points=80,
        dwell_step_mm=5.0,
        min_entry_separation_mm=args.min_entry_sep_mm,
        min_path_separation_mm=2.0,
        score_hrctv=not args.fast,
        oar_clearance_mm=oar_clearance_mm,
        axis_hint_zyx=rtplan_tandem,
        debug_rejections=args.debug_rejections,
        return_entry_plane=True,
    )
    vagina_label = label_mapping.get("Vagina")
    vagina_vox = np.argwhere(structure_mask == vagina_label) if vagina_label is not None else None
    pca_axis = _pca_major_axis(vagina_vox) if vagina_vox is not None and vagina_vox.size else None

    env = BrachyRL_TG43(
        structure_mask=structure_mask,
        max_needles=10,
        anatomical_library=anatomical_library,
        voxel_spacing_mm=voxel_spacing,
    )

    visualize_bent_needles(
        structure_mask=structure_mask,
        label_mapping=label_mapping,
        anatomical_library=anatomical_library,
        affine=affine,
        voxel_spacing=voxel_spacing,
        tandem_paths=tandem_paths,
        ovoid_paths=ovoid_paths,
        entry_plane=entry_plane,
        show_entry_debug=args.debug_entry_plane,
        pca_axis=pca_axis if args.show_pca_axis else None,
        screenshot_path=args.screenshot,
        save_html=args.save_html,
        save_vtk=args.save_vtk,
    )
