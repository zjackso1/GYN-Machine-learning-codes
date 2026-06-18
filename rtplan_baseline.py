from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None

from env.reward_logic import (
    AIR_KERMA_STRENGTH_U,
    deposit_line_superposition,
    prepare_structure_coords,
)


def find_rtplan(rtstruct_path: Optional[str]) -> Optional[str]:
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


def _channel_label(channel) -> str:
    for key in ("ChannelName", "ChannelID", "ChannelDescription", "SourceApplicatorID"):
        val = getattr(channel, key, None)
        if val:
            return str(val)
    return "Unnamed"


def _get_cp_time(cp) -> Optional[float]:
    for key in ("CumulativeTimeWeight", "ControlPointRelativeTime", "ControlPointTime"):
        if hasattr(cp, key):
            try:
                return float(getattr(cp, key))
            except Exception:
                continue
    return None


def _extract_dwell_positions_times(channel, tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    seq = None
    if hasattr(channel, "BrachyControlPointSequence"):
        seq = channel.BrachyControlPointSequence
    elif hasattr(channel, "ControlPointSequence"):
        seq = channel.ControlPointSequence
    if not seq:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    positions = []
    times = []
    for cp in seq:
        pos = getattr(cp, "ControlPoint3DPosition", None)
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        if arr.size != 3:
            continue
        positions.append(arr)
        t = _get_cp_time(cp)
        times.append(np.nan if t is None else float(t))

    if not positions:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    positions = np.stack(positions, axis=0)
    times = np.asarray(times, dtype=np.float32)

    total_time = getattr(channel, "ChannelTotalTime", None)
    if total_time is not None:
        try:
            total_time = float(total_time)
        except Exception:
            total_time = None

    if np.all(~np.isfinite(times)):
        times = np.zeros(len(positions), dtype=np.float32)
    if total_time is not None and total_time > 0.0:
        max_time = np.nanmax(times)
        if max_time <= 1.0 + 1e-6:
            times = times * float(total_time)

    dwell_pos = []
    dwell_times = []
    i = 0
    while i < len(positions) - 1:
        p0 = positions[i]
        p1 = positions[i + 1]
        t0 = times[i]
        t1 = times[i + 1]
        if np.linalg.norm(p0 - p1) <= tol:
            dt = 0.0 if not np.isfinite(t0) or not np.isfinite(t1) else max(0.0, t1 - t0)
            dwell_pos.append(p0)
            dwell_times.append(dt)
            i += 2
        else:
            dt = 0.0 if not np.isfinite(t0) or not np.isfinite(t1) else max(0.0, t1 - t0)
            dwell_pos.append(p0)
            dwell_times.append(dt)
            i += 1

    if not dwell_pos:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    dwell_pos = np.stack(dwell_pos, axis=0).astype(np.float32)
    dwell_times = np.asarray(dwell_times, dtype=np.float32)

    if not np.any(dwell_times > 0.0):
        if total_time is not None and total_time > 0.0:
            dwell_times = np.full(len(dwell_pos), float(total_time) / len(dwell_pos), dtype=np.float32)
        else:
            dwell_times = np.ones(len(dwell_pos), dtype=np.float32)

    return dwell_pos, dwell_times


def _match_channels(infos, keywords: Iterable[str]) -> list:
    needles = [s.lower() for s in keywords if s]
    if not needles:
        return []
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
    seen = set()
    unique = []
    for info in selected:
        idx = info["index"]
        if idx in seen:
            continue
        seen.add(idx)
        unique.append(info)
    return unique


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


def _filter_inbounds(points_zyx: np.ndarray, shape, return_mask: bool = False):
    if points_zyx.size == 0:
        return (points_zyx, np.zeros((0,), dtype=bool)) if return_mask else points_zyx
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
    return (points_zyx, keep) if return_mask else points_zyx


def _select_tandem_ovoid_channels(infos: list[dict]) -> Tuple[Optional[dict], list[dict]]:
    tandem_matches = _match_channels(
        infos,
        ["tandem", "applicator1", "applicator 1", "applicator-1"],
    )
    ovoid_matches = _match_channels(
        infos,
        ["ovoid", "applicator2", "applicator 2", "applicator-2",
         "applicator3", "applicator 3", "applicator-3"],
    )

    if len(tandem_matches) == 1:
        tandem = tandem_matches[0]
    elif len(infos) == 1:
        tandem = infos[0]
    else:
        ordered = sorted(infos, key=lambda d: (d.get("number") is None, d.get("number", d["index"])))
        tandem = ordered[0] if ordered else None
        if not ovoid_matches and len(ordered) >= 3:
            ovoid_matches = ordered[1:3]

    if tandem is not None and not ovoid_matches:
        remaining = [info for info in infos if info is not tandem]
        ordered = sorted(remaining, key=lambda d: (d.get("number") is None, d.get("number", d["index"])))
        ovoid_matches = ordered[:2]

    return tandem, ovoid_matches


def _extract_air_kerma_strength(ds, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    if pydicom is None:
        return float(AIR_KERMA_STRENGTH_U)
    for elem in ds.iterall():
        if elem.keyword in ("ReferenceAirKermaRate", "AirKermaStrength"):
            try:
                return float(elem.value)
            except Exception:
                continue
    return float(AIR_KERMA_STRENGTH_U)


def compute_rtplan_baseline_dose(
    rtplan_path: str,
    ct_series: str,
    structure_mask: np.ndarray,
    label_mapping: dict,
    voxel_spacing: tuple[float, float, float],
    dose_model: str = "blend",
    air_kerma_strength: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], list[np.ndarray]]:
    if pydicom is None:
        raise RuntimeError("pydicom is required to compute RTPLAN baseline.")

    ds = pydicom.dcmread(rtplan_path, stop_before_pixels=True)
    channels = _collect_rtplan_channels(ds)
    if not channels:
        raise ValueError("No ChannelSequence found in RTPLAN.")

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

    tandem_info, ovoid_infos = _select_tandem_ovoid_channels(infos)
    if tandem_info is None:
        labels = [f"[{m['index']}] {_channel_label(m['channel'])}" for m in infos]
        raise RuntimeError(f"Could not identify tandem channel in RTPLAN. Channels={labels}")

    ct_geom = _load_ct_geometry(ct_series)
    strength = _extract_air_kerma_strength(ds, air_kerma_strength)

    coords_vox, coords_mm = prepare_structure_coords(
        structure_mask,
        label_mapping,
        voxel_spacing,
        include_structures=list(label_mapping.keys()),
    )

    dose_map = np.zeros_like(structure_mask, dtype=np.float32)

    model = str(dose_model).lower()
    if model == "kernel":
        model = "line"

    def add_channel(ch) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pos_mm, dwell_times = _extract_dwell_positions_times(ch)
        if pos_mm.size == 0:
            return None
        pos_zyx = _rtplan_points_to_vox_zyx(pos_mm, ct_geom)
        pos_zyx, keep = _filter_inbounds(pos_zyx, structure_mask.shape, return_mask=True)
        if pos_zyx.size == 0:
            return None
        if dwell_times.shape[0] == keep.shape[0]:
            dwell_times = dwell_times[keep]
        if pos_zyx.shape[0] != dwell_times.shape[0]:
            dwell_times = dwell_times[:pos_zyx.shape[0]]
        delta = deposit_line_superposition(
            pos_zyx,
            dwell_times,
            structure_mask,
            voxel_spacing,
            label_mapping,
            include_structures=list(label_mapping.keys()),
            air_kerma_strength=strength,
            structure_indices=coords_vox,
            precomputed_coords=coords_mm,
            dose_model=model,
        )
        nonlocal dose_map
        dose_map += delta
        return pos_zyx, dwell_times

    tandem_path = None
    ovoid_paths = []
    add_res = add_channel(tandem_info["channel"])
    if add_res is not None:
        tandem_path = add_res[0]

    for info in ovoid_infos:
        add_res = add_channel(info["channel"])
        if add_res is not None:
            ovoid_paths.append(add_res[0])

    return dose_map, tandem_path, ovoid_paths


def load_rtplan_paths(
    rtplan_path: str,
    ct_series: str,
    structure_mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], list[np.ndarray]]:
    if pydicom is None:
        raise RuntimeError("pydicom is required to load RTPLAN paths.")

    ds = pydicom.dcmread(rtplan_path, stop_before_pixels=True)
    channels = _collect_rtplan_channels(ds)
    if not channels:
        raise ValueError("No ChannelSequence found in RTPLAN.")

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

    tandem_info, ovoid_infos = _select_tandem_ovoid_channels(infos)
    if tandem_info is None:
        labels = [f"[{m['index']}] {_channel_label(m['channel'])}" for m in infos]
        raise RuntimeError(f"Could not identify tandem channel in RTPLAN. Channels={labels}")

    ct_geom = _load_ct_geometry(ct_series)

    def _channel_to_path(ch) -> Optional[np.ndarray]:
        pos_mm, _times = _extract_dwell_positions_times(ch)
        if pos_mm.size == 0:
            return None
        pos_zyx = _rtplan_points_to_vox_zyx(pos_mm, ct_geom)
        pos_zyx, _keep = _filter_inbounds(pos_zyx, structure_mask.shape, return_mask=True)
        if pos_zyx.size == 0:
            return None
        return pos_zyx

    tandem_path = _channel_to_path(tandem_info["channel"])
    ovoid_paths = []
    for info in ovoid_infos:
        path = _channel_to_path(info["channel"])
        if path is not None:
            ovoid_paths.append(path)

    return tandem_path, ovoid_paths
