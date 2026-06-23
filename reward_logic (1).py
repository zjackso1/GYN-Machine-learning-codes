import numpy as np
from numba import njit

from env import tg43

# ---------------- Labels & Constraints ----------------
HRCTV_LABEL = 1  # HRCTV label in the structure mask

LABELS = {
    "HRCTV": 1,
    "Rectum": 2,
    "Bladder": 3,
    "Sigmoid": 4,
    "Bowel": 5,
    "Vagina": 6
}

OAR_CONSTRAINTS = {
    "Rectum": 390.0,
    "Bladder": 440.0,
    "Sigmoid": 390.0,
    "Bowel": 340.0,
    "Vagina": 680.0
}

MISS_PENALTY_WEIGHT = 1.0  # Penalty for needles missing HRCTV
TRAINING_DOSE_SCALE = 10.0  # calibrated boost for kernel mode (if used)
AIR_KERMA_STRENGTH_U = 40700.0  # default Ir-192 source strength
OAR_PENALTY_WEIGHTS = {
    "Rectum": 4.0,
    "Bladder": 8.0,
    "Sigmoid": 8.0,
    "Bowel": 4.0,
    "Vagina": 22.0,
}

# Hard cap for HRCTV D90 (cGy). Plans above this should be discouraged.
D90_MAX_CGY = 800.0
# Quadratic penalty weight on overshoot beyond D90_MAX_CGY.
D90_OVERSHOOT_PENALTY = 0.10
# Target band for HRCTV D90 (cGy).
D90_TARGET_CGY = 650.0
D90_TOL_CGY = 50.0

# TG-43 blend model defaults (validated against TPS phantom)
TG43_BLEND_CENTER_DEG = 90.0
TG43_BLEND_WIDTH_DEG = 90.0
TG43_FLIP_ANISOTROPY = True
TG43_CABLE_ANISO_MIN = 0.55
TG43_CABLE_ANISO_THETA_DEG = 150.0

def voxels_to_mm(points_zyx, voxel_spacing_mm):
    arr = np.asarray(points_zyx, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    dz, dy, dx = voxel_spacing_mm
    x_mm = arr[:, 2] * dx
    y_mm = arr[:, 1] * dy
    z_mm = arr[:, 0] * dz
    return np.stack([x_mm, y_mm, z_mm], axis=1)

MIN_RADIUS_CM = 0.2
RADIAL_GRID = tg43.RADIAL_DATA_CM.astype(np.float32)
# Use TG-43 normalized radial dose function (g_L(1 cm) = 1.0).
RADIAL_VALS = np.array([tg43.radial_g(r) for r in tg43.RADIAL_DATA_CM], dtype=np.float32)
ANISO_RADIAL_GRID = tg43.ANISO_RADIAL_CM.astype(np.float32)
ANISO_THETA_GRID = tg43.ANISO_THETA_DEG.astype(np.float32)
ANISO_TABLE = tg43.ANISO_VALUES.astype(np.float32)

@njit(cache=True)
def _interp_log_linear(x, xp, fp):
    # Log-linear interpolation: exp(interp(x, xp, log(fp)))
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    for i in range(xp.size - 1):
        x0 = xp[i]
        x1 = xp[i + 1]
        if x <= x1:
            if x1 == x0:
                return fp[i]
            t = (x - x0) / (x1 - x0)
            y0 = fp[i]
            y1 = fp[i + 1]
            if y0 <= 0.0 or y1 <= 0.0:
                return y0 + t * (y1 - y0)
            return np.exp(np.log(y0) + t * (np.log(y1) - np.log(y0)))
    return fp[-1]


@njit(cache=True)
def _anisotropy_factor(r_cm, theta_deg, radial_grid, theta_grid, table):
    r = min(max(r_cm, radial_grid[0]), radial_grid[-1])
    theta = min(max(theta_deg, theta_grid[0]), theta_grid[-1])

    r_idx = radial_grid.size - 2
    for i in range(radial_grid.size - 1):
        if r <= radial_grid[i + 1]:
            r_idx = i
            break
    t_idx = theta_grid.size - 2
    for j in range(theta_grid.size - 1):
        if theta <= theta_grid[j + 1]:
            t_idx = j
            break

    r0 = radial_grid[r_idx]
    r1 = radial_grid[r_idx + 1]
    rt = 0.0 if r1 == r0 else (r - r0) / (r1 - r0)

    t0 = theta_grid[t_idx]
    t1 = theta_grid[t_idx + 1]
    tt = 0.0 if t1 == t0 else (theta - t0) / (t1 - t0)

    f00 = table[t_idx, r_idx]
    f01 = table[t_idx, r_idx + 1]
    f10 = table[t_idx + 1, r_idx]
    f11 = table[t_idx + 1, r_idx + 1]

    f0 = f00 + rt * (f01 - f00)
    f1 = f10 + rt * (f11 - f10)
    return f0 + tt * (f1 - f0)


@njit(cache=True)
def _smoothstep01(t):
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def _geometry_factor_line_scalar(rho_cm, z_cm, length_cm):
    # TG-43U1 line-source geometry factor using r, theta expression.
    rho = abs(rho_cm)
    r_cm = np.sqrt(rho * rho + z_cm * z_cm)
    if r_cm <= 1e-12:
        return 0.0
    cos_theta = z_cm / max(r_cm, 1e-12)
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-10:
        denom = (r_cm * r_cm) - (length_cm * length_cm * 0.25)
        if denom < 1e-12:
            denom = 1e-12
        return 1.0 / denom

    num1 = r_cm * np.cos(theta) - length_cm / 2.0
    den1 = np.sqrt(r_cm * r_cm + (length_cm / 2.0) ** 2 - length_cm * r_cm * np.cos(theta))
    num2 = r_cm * np.cos(theta) + length_cm / 2.0
    den2 = np.sqrt(r_cm * r_cm + (length_cm / 2.0) ** 2 + length_cm * r_cm * np.cos(theta))

    den1 = max(den1, 1e-12)
    den2 = max(den2, 1e-12)
    term1 = np.arccos(min(1.0, max(-1.0, num1 / den1)))
    term2 = np.arccos(min(1.0, max(-1.0, num2 / den2)))
    return (term1 - term2) / (length_cm * r_cm * sin_theta)


@njit(cache=True)
def _line_dose_single_structure(coords_mm, dwell_mm, tangents_mm, weights, air_kerma,
                                ref_geom, ref_radial, dose_rate_const, source_len,
                                radial_grid, radial_vals, anis_r, anis_theta, anis_vals):
    n_vox = coords_mm.shape[0]
    contributions = np.zeros(n_vox, dtype=np.float32)
    # g_L is normalized to 1.0 at 1 cm; normalize by geometry only.
    norm_ref = ref_geom
    deg_per_rad = 180.0 / np.pi

    for d_idx in range(dwell_mm.shape[0]):
        w = weights[d_idx]
        if w <= 0.0:
            continue
        dwell = dwell_mm[d_idx]
        axis = tangents_mm[d_idx]
        ax0 = axis[0]
        ax1 = axis[1]
        ax2 = axis[2]
        for v in range(n_vox):
            diff0 = coords_mm[v, 0] - dwell[0]
            diff1 = coords_mm[v, 1] - dwell[1]
            diff2 = coords_mm[v, 2] - dwell[2]
            proj = diff0 * ax0 + diff1 * ax1 + diff2 * ax2
            diff_sq = diff0 * diff0 + diff1 * diff1 + diff2 * diff2
            perp_sq = diff_sq - proj * proj
            if perp_sq < 0.0:
                perp_sq = 0.0
            rho_cm = np.sqrt(perp_sq) / 10.0
            z_cm = proj / 10.0
            r_cm = np.sqrt(rho_cm * rho_cm + z_cm * z_cm)
            if r_cm < 0.2:
                r_cm = 0.2

            theta_arg = z_cm / max(r_cm, 1e-6)
            if theta_arg < -1.0:
                theta_arg = -1.0
            elif theta_arg > 1.0:
                theta_arg = 1.0
            theta_deg = np.arccos(theta_arg) * deg_per_rad

            geom = _geometry_factor_line_scalar(rho_cm, z_cm, source_len)
            radial = _interp_log_linear(r_cm, radial_grid, radial_vals)
            anis = _anisotropy_factor(r_cm, theta_deg, anis_r, anis_theta, anis_vals)
            dose_rate = dose_rate_const * (geom * radial * anis) / norm_ref
            contributions[v] += air_kerma * (w / 3600.0) * dose_rate
    return contributions


@njit(cache=True)
def _blend_dose_single_structure(
    coords_mm,
    dwell_mm,
    tangents_mm,
    weights,
    air_kerma,
    ref_geom,
    ref_radial,
    dose_rate_const,
    source_len,
    ref_r_cm,
    radial_grid,
    radial_vals,
    anis_r,
    anis_theta,
    anis_vals,
    blend_center_deg,
    blend_width_deg,
    flip_anisotropy,
    cable_aniso_min,
    cable_aniso_theta_deg,
):
    n_vox = coords_mm.shape[0]
    contributions = np.zeros(n_vox, dtype=np.float32)
    # g_L is normalized to 1.0 at 1 cm; normalize by geometry only.
    line_norm = ref_geom
    point_norm = 1.0 / (ref_r_cm * ref_r_cm)
    deg_per_rad = 180.0 / np.pi
    half_width = max(1e-6, blend_width_deg * 0.5)

    for d_idx in range(dwell_mm.shape[0]):
        w = weights[d_idx]
        if w <= 0.0:
            continue
        dwell = dwell_mm[d_idx]
        axis = tangents_mm[d_idx]
        ax0 = axis[0]
        ax1 = axis[1]
        ax2 = axis[2]
        for v in range(n_vox):
            diff0 = coords_mm[v, 0] - dwell[0]
            diff1 = coords_mm[v, 1] - dwell[1]
            diff2 = coords_mm[v, 2] - dwell[2]
            proj = diff0 * ax0 + diff1 * ax1 + diff2 * ax2
            diff_sq = diff0 * diff0 + diff1 * diff1 + diff2 * diff2
            perp_sq = diff_sq - proj * proj
            if perp_sq < 0.0:
                perp_sq = 0.0
            rho_cm = np.sqrt(perp_sq) / 10.0
            z_cm = proj / 10.0
            r_line = np.sqrt(rho_cm * rho_cm + z_cm * z_cm)
            if r_line < MIN_RADIUS_CM:
                r_line = MIN_RADIUS_CM
            r_point = np.sqrt(diff_sq) / 10.0
            if r_point < MIN_RADIUS_CM:
                r_point = MIN_RADIUS_CM

            theta_arg = z_cm / max(r_line, 1e-6)
            if theta_arg < -1.0:
                theta_arg = -1.0
            elif theta_arg > 1.0:
                theta_arg = 1.0
            theta_deg = np.arccos(theta_arg) * deg_per_rad
            if flip_anisotropy > 0.5:
                theta_deg = 180.0 - theta_deg

            theta_arg_p = z_cm / max(r_point, 1e-6)
            if theta_arg_p < -1.0:
                theta_arg_p = -1.0
            elif theta_arg_p > 1.0:
                theta_arg_p = 1.0
            theta_deg_p = np.arccos(theta_arg_p) * deg_per_rad
            if flip_anisotropy > 0.5:
                theta_deg_p = 180.0 - theta_deg_p

            geom_line = _geometry_factor_line_scalar(rho_cm, z_cm, source_len)
            radial_line = _interp_log_linear(r_line, radial_grid, radial_vals)
            anis_line = _anisotropy_factor(r_line, theta_deg, anis_r, anis_theta, anis_vals)
            if cable_aniso_min > 0.0 and theta_deg >= cable_aniso_theta_deg and anis_line < cable_aniso_min:
                anis_line = cable_aniso_min

            geom_point = 1.0 / (r_point * r_point)
            radial_point = _interp_log_linear(r_point, radial_grid, radial_vals)
            anis_point = _anisotropy_factor(r_point, theta_deg_p, anis_r, anis_theta, anis_vals)
            if cable_aniso_min > 0.0 and theta_deg_p >= cable_aniso_theta_deg and anis_point < cable_aniso_min:
                anis_point = cable_aniso_min

            dist = theta_deg - blend_center_deg
            if dist < 0.0:
                dist = -dist
            t = dist / half_width
            alpha = _smoothstep01(t)

            line_term = (geom_line * radial_line * anis_line) / line_norm
            point_term = (geom_point * radial_point * anis_point) / point_norm
            dose_rate = dose_rate_const * ((1.0 - alpha) * line_term + alpha * point_term)
            contributions[v] += air_kerma * (w / 3600.0) * dose_rate
    return contributions


def _line_dose_numba_single(coords_mm, dwell_mm, tangents, weights, air_kerma_strength):
    coords32 = np.ascontiguousarray(coords_mm, dtype=np.float32)
    dwell32 = np.ascontiguousarray(dwell_mm, dtype=np.float32)
    tangents32 = np.ascontiguousarray(tangents, dtype=np.float32)
    weights32 = np.ascontiguousarray(weights, dtype=np.float32)
    return _line_dose_single_structure(
        coords32,
        dwell32,
        tangents32,
        weights32,
        np.float32(air_kerma_strength),
        np.float32(tg43.REF_GEOM),
        np.float32(tg43.REF_RADIAL),
        np.float32(tg43.DOSE_RATE_CONSTANT),
        np.float32(tg43.SOURCE_LENGTH_CM),
        RADIAL_GRID,
        RADIAL_VALS,
        ANISO_RADIAL_GRID,
        ANISO_THETA_GRID,
        ANISO_TABLE,
    )


def _blend_dose_numba_single(
    coords_mm,
    dwell_mm,
    tangents,
    weights,
    air_kerma_strength,
    blend_center_deg=TG43_BLEND_CENTER_DEG,
    blend_width_deg=TG43_BLEND_WIDTH_DEG,
    flip_anisotropy=TG43_FLIP_ANISOTROPY,
    cable_aniso_min=TG43_CABLE_ANISO_MIN,
    cable_aniso_theta_deg=TG43_CABLE_ANISO_THETA_DEG,
):
    coords32 = np.ascontiguousarray(coords_mm, dtype=np.float32)
    dwell32 = np.ascontiguousarray(dwell_mm, dtype=np.float32)
    tangents32 = np.ascontiguousarray(tangents, dtype=np.float32)
    weights32 = np.ascontiguousarray(weights, dtype=np.float32)
    return _blend_dose_single_structure(
        coords32,
        dwell32,
        tangents32,
        weights32,
        np.float32(air_kerma_strength),
        np.float32(tg43.REF_GEOM),
        np.float32(tg43.REF_RADIAL),
        np.float32(tg43.DOSE_RATE_CONSTANT),
        np.float32(tg43.SOURCE_LENGTH_CM),
        np.float32(tg43.REF_R_CM),
        RADIAL_GRID,
        RADIAL_VALS,
        ANISO_RADIAL_GRID,
        ANISO_THETA_GRID,
        ANISO_TABLE,
        np.float32(blend_center_deg),
        np.float32(blend_width_deg),
        np.float32(1.0 if flip_anisotropy else 0.0),
        np.float32(0.0 if cable_aniso_min is None else cable_aniso_min),
        np.float32(cable_aniso_theta_deg),
    )

def radial_g_vector(r_cm_array):
    vals = np.asarray(r_cm_array, dtype=np.float32)
    vals = np.maximum(vals, MIN_RADIUS_CM)
    rmax = float(tg43.RADIAL_DATA_CM[-1])
    out = np.interp(np.minimum(vals, rmax), tg43.RADIAL_DATA_CM, tg43.RADIAL_G)
    mask = vals > rmax
    if np.any(mask):
        x1 = float(tg43.RADIAL_DATA_CM[-2])
        x2 = float(tg43.RADIAL_DATA_CM[-1])
        y1 = float(tg43.RADIAL_G[-2])
        y2 = float(tg43.RADIAL_G[-1])
        if x2 > x1 and y1 > 0.0 and y2 > 0.0:
            slope = (np.log(y2) - np.log(y1)) / (x2 - x1)
            out[mask] = np.exp(np.log(y2) + slope * (vals[mask] - x2))
        else:
            out[mask] = y2
    # Normalize to g_L(1 cm) = 1.0.
    return out / float(tg43.RADIAL_G_REF)

def prepare_structure_coords(
    structure_mask,
    label_mapping,
    voxel_spacing_mm,
    include_structures=None,
    max_voxels_per_structure=None,
    rng_seed=0,
):
    if include_structures is None:
        include_structures = list(label_mapping.keys())
    if max_voxels_per_structure is None:
        max_voxels_per_structure = {}

    rng = np.random.default_rng(rng_seed)
    indices = {}
    coords = {}
    for name in include_structures:
        label = label_mapping.get(name)
        if label is None:
            continue
        idx = np.argwhere(structure_mask == label)
        if idx.size == 0:
            continue
        cap = max_voxels_per_structure.get(name)
        if cap is not None and idx.shape[0] > cap:
            keep = rng.choice(idx.shape[0], cap, replace=False)
            idx = idx[keep]
        indices[label] = idx
        coords[label] = voxels_to_mm(idx, voxel_spacing_mm)
    return indices, coords

def smooth_dose_map(dose_map, kernel_size=3):
    if kernel_size <= 1:
        return dose_map
    pad = kernel_size // 2
    padded = np.pad(dose_map, pad_width=pad, mode="edge")
    smoothed = np.zeros_like(dose_map, dtype=np.float32)
    for dz in range(kernel_size):
        z_start = dz
        z_end = z_start + dose_map.shape[0]
        for dy in range(kernel_size):
            y_start = dy
            y_end = y_start + dose_map.shape[1]
            for dx in range(kernel_size):
                x_start = dx
                x_end = x_start + dose_map.shape[2]
                smoothed += padded[z_start:z_end, y_start:y_end, x_start:x_end]
    smoothed /= float(kernel_size ** 3)
    return smoothed


def compute_path_tangents_mm(points_mm):
    pts = np.asarray(points_mm, dtype=np.float32)
    n = pts.shape[0]
    tangents = np.zeros_like(pts, dtype=np.float32)
    if n == 0:
        return tangents
    if n == 1:
        tangents[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return tangents
    for i in range(n):
        if i == 0:
            vec = pts[1] - pts[0]
        elif i == n - 1:
            vec = pts[-1] - pts[-2]
        else:
            vec = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            tangents[i] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            tangents[i] = (vec / norm).astype(np.float32)
    return tangents


def deposit_line_superposition(
    needle_positions,
    weights,
    structure_mask,
    voxel_spacing_mm,
    label_mapping,
    include_structures=None,
    air_kerma_strength=AIR_KERMA_STRENGTH_U,
    structure_indices=None,
    precomputed_coords=None,
    smoothing_kernel=1,
    path_tangents=None,
    dose_model="line",
    blend_center_deg=TG43_BLEND_CENTER_DEG,
    blend_width_deg=TG43_BLEND_WIDTH_DEG,
    flip_anisotropy=TG43_FLIP_ANISOTROPY,
    cable_aniso_min=TG43_CABLE_ANISO_MIN,
    cable_aniso_theta_deg=TG43_CABLE_ANISO_THETA_DEG,
):
    positions = np.asarray(needle_positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("needle_positions must be (N,3) array of voxels")

    if weights is None:
        weights = np.ones(len(positions), dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] != positions.shape[0]:
            raise ValueError("weights must match number of dwell positions")

    if include_structures is None:
        include_structures = list(label_mapping.keys())

    if precomputed_coords is not None and structure_indices is not None:
        coords_mm = precomputed_coords
        coords_vox = structure_indices
    else:
        coords_vox, coords_mm = prepare_structure_coords(
            structure_mask,
            label_mapping,
            voxel_spacing_mm,
            include_structures=include_structures,
        )

    dwell_mm = voxels_to_mm(positions, voxel_spacing_mm)
    if path_tangents is None:
        path_tangents = compute_path_tangents_mm(dwell_mm)

    model = str(dose_model).lower()
    if model not in {"line", "blend"}:
        raise ValueError("dose_model must be 'line' or 'blend'")

    dose_map = np.zeros_like(structure_mask, dtype=np.float32)
    for label, idx in coords_vox.items():
        coord_mm = coords_mm[label]
        if model == "blend":
            contributions = _blend_dose_numba_single(
                coord_mm,
                dwell_mm,
                path_tangents,
                weights,
                air_kerma_strength,
                blend_center_deg=blend_center_deg,
                blend_width_deg=blend_width_deg,
                flip_anisotropy=flip_anisotropy,
                cable_aniso_min=cable_aniso_min,
                cable_aniso_theta_deg=cable_aniso_theta_deg,
            )
        else:
            contributions = _line_dose_numba_single(
                coord_mm,
                dwell_mm,
                path_tangents,
                weights,
                air_kerma_strength,
            )
        dose_map[idx[:, 0], idx[:, 1], idx[:, 2]] += contributions
    if smoothing_kernel and smoothing_kernel > 1:
        dose_map = smooth_dose_map(dose_map, kernel_size=int(smoothing_kernel))
    return dose_map

def add_far_field_tail(
    dose_map,
    needle_positions,
    weights,
    structure_indices,
    structure_coords_mm,
    voxel_spacing_mm,
    radius_mm,
    include_labels=None,
    air_kerma_strength=AIR_KERMA_STRENGTH_U,
    path_tangents=None,
):
    if include_labels is None:
        include_labels = list(structure_indices.keys())

    tail_buffers = {label: np.zeros(len(structure_indices[label]), dtype=np.float32)
                    for label in include_labels if label in structure_indices}
    if not tail_buffers:
        return

    dwell_mm = voxels_to_mm(needle_positions, voxel_spacing_mm)
    if path_tangents is None:
        path_tangents = compute_path_tangents_mm(dwell_mm)
    radius_mm = float(radius_mm)

    for dwell, axis, w in zip(dwell_mm, path_tangents, weights):
        if w <= 0.0:
            continue
        for label in include_labels:
            if label not in tail_buffers:
                continue
            coords_mm = structure_coords_mm[label]
            diff = coords_mm - dwell[None, :]
            proj = diff @ axis
            perp = diff - np.outer(proj, axis)
            r_mm = np.linalg.norm(perp, axis=1)
            mask = r_mm > radius_mm
            if not np.any(mask):
                continue
            rho_cm = r_mm[mask] / 10.0
            z_cm = proj[mask] / 10.0
            r_cm = np.sqrt(rho_cm ** 2 + z_cm ** 2)
            theta = np.arccos(np.clip(z_cm / np.maximum(r_cm, 1e-6), -1.0, 1.0))
            dose_rate = tg43.dose_rate_per_unit_strength(r_cm, rho_cm, z_cm, theta)
            tail_buffers[label][mask] += (air_kerma_strength * (w / 3600.0)) * dose_rate.astype(np.float32)

    for label, buf in tail_buffers.items():
        if not np.any(buf):
            continue
        idx = structure_indices[label]
        dose_map[idx[:, 0], idx[:, 1], idx[:, 2]] += buf

    # ---------------- Dose Metrics ----------------
def compute_dose_metrics(dose_map, structure_mask, voxel_spacing_mm=None, structure_flat_indices=None):
    flat = None
    hrctv_values = None
    if structure_flat_indices is not None:
        idx = structure_flat_indices.get(HRCTV_LABEL) if isinstance(structure_flat_indices, dict) else None
        if idx is not None and len(idx):
            flat = dose_map.ravel()
            hrctv_values = flat[idx]
    if hrctv_values is None:
        hrctv_mask = (structure_mask == HRCTV_LABEL)
        hrctv_values = dose_map[hrctv_mask]
    if hrctv_values.size > 0:
        hrctv_d90 = np.percentile(hrctv_values, 10)
        hrctv_d98 = np.percentile(hrctv_values, 2)
        hrctv_mean = float(hrctv_values.mean())
    else:
        hrctv_d90 = 0.0
        hrctv_d98 = 0.0
        hrctv_mean = 0.0

    def top_k_mean(arr, k):
        if arr.size == 0:
            return 0.0
        if arr.size <= k:
            return arr.mean()
        return np.partition(arr, -k)[-k:].mean()

    voxels_2cc = 2000
    if voxel_spacing_mm is not None:
        dz, dy, dx = voxel_spacing_mm
        voxel_volume_cc = (dz * dy * dx) / 1000.0
        if voxel_volume_cc > 0:
            voxels_2cc = max(1, int(np.ceil(2.0 / voxel_volume_cc)))

    def d2cc(label):
        arr = None
        if structure_flat_indices is not None and isinstance(structure_flat_indices, dict):
            idx = structure_flat_indices.get(label)
            if idx is not None and len(idx):
                nonlocal flat
                if flat is None:
                    flat = dose_map.ravel()
                arr = flat[idx]
        if arr is None:
            arr = dose_map[structure_mask == label]
        if arr.size == 0:
            return 0.0
        k = min(voxels_2cc, arr.size)
        return top_k_mean(arr, k)

    rectum_d2cc = d2cc(LABELS["Rectum"])
    bladder_d2cc = d2cc(LABELS["Bladder"])
    sigmoid_d2cc = d2cc(LABELS["Sigmoid"])
    bowel_d2cc = d2cc(LABELS["Bowel"])
    vagina_d2cc = d2cc(LABELS["Vagina"])

    return (
        hrctv_d90,
        hrctv_d98,
        hrctv_mean,
        rectum_d2cc,
        bladder_d2cc,
        sigmoid_d2cc,
        bowel_d2cc,
        vagina_d2cc,
    )


def compute_hrctv_coverage(dose_map, structure_mask, thresholds=(200.0, 400.0, 600.0), structure_flat_indices=None):
    hrctv_values = None
    if structure_flat_indices is not None and isinstance(structure_flat_indices, dict):
        idx = structure_flat_indices.get(HRCTV_LABEL)
        if idx is not None and len(idx):
            hrctv_values = dose_map.ravel()[idx]
    if hrctv_values is None:
        hrctv_values = dose_map[structure_mask == HRCTV_LABEL]
    coverages = []
    if hrctv_values.size == 0:
        return [0.0 for _ in thresholds]
    for thr in thresholds:
        coverages.append(float((hrctv_values >= thr).mean()))
    return coverages

# ---------------- Reward Function ----------------
def compute_reward(
    hrctv_d90,
    hrctv_d98,
    hrctv_mean,
    coverage_200,
    coverage_400,
    coverage_600,
    oar_doses,
    baseline_oar_doses,
    needle_positions,
    delta_d90,
    total_dwell_time=None,
    
    penalty=0.0,
    stop=False
):
    """
    Reward logic for straight-needle PPO + STOP action.

    Terms:
      • HRCTV term: reward increases as D90 reaches Rx (presumed 700–800 cGy)
      • OAR term: penalize D2cc beyond soft limits (200 cGy default)
      • Dwell penalty: discourage excessive total dwell once D90 is in-band
      • Needle penalty: discourage unnecessary needles
      • STOP penalty: if STOP but HRCTV underdosed → penalize
    """

    # -----------------------------
    # 1) HRCTV COVERAGE TERM
    # -----------------------------
    # Normalize D90 to roughly 0–1 range for PPO
    # (adjust target if you use a different Rx)
    target_d90 = D90_TARGET_CGY
    target_d98 = 450.0
    d90_tol = D90_TOL_CGY
    d98_tol = 50.0
    sigma90 = d90_tol / np.sqrt(2 * np.log(2))
    sigma98 = d98_tol / np.sqrt(2 * np.log(2))
    d90_low = target_d90 - d90_tol
    d90_high = target_d90 + d90_tol

    # Symmetric Gaussian terms (kept for diagnostics; not used in lexicographic reward)
    d90_term = np.exp(-0.5 * ((hrctv_d90 - target_d90) / sigma90) ** 2)
    d98_term = np.exp(-0.5 * ((hrctv_d98 - target_d98) / sigma98) ** 2)
    target_mean = 650.0
    mean_term = np.clip(hrctv_mean / max(target_mean, 1e-6), 0, 1.0)

    zero_penalty = 0.0
    if hrctv_d90 < 1.0:
        zero_penalty = 10.0

    d90_overshoot_penalty = 0.0
    if hrctv_d90 > D90_MAX_CGY:
        over_cgy = float(hrctv_d90 - D90_MAX_CGY)
        d90_overshoot_penalty = D90_OVERSHOOT_PENALTY * (over_cgy ** 2)
    d90_excess_penalty = 0.0
    if hrctv_d90 > d90_high:
        d90_excess_penalty = 0.02 * (float(hrctv_d90 - d90_high) ** 2)
    d98_excess_penalty = 0.0
    if hrctv_d98 > (target_d98 + d98_tol):
        d98_excess_penalty = 0.01 * (float(hrctv_d98 - (target_d98 + d98_tol)) ** 2)


    # -----------------------------
    # 2) OAR PENALTY
    # -----------------------------
    oar_penalty = 0.0
    oar_sparing = 0.0
    oar_ok = True
    oar_over_max = 0.0
    use_baseline = isinstance(baseline_oar_doses, dict) and len(baseline_oar_doses) > 0
    for organ, dose in oar_doses.items():
        limit = float(OAR_CONSTRAINTS.get(organ, 200.0))
        weight = OAR_PENALTY_WEIGHTS.get(organ, 1.0)
        if limit > 0:
            frac = float(dose) / float(limit)
            if frac < 1.0:
                oar_sparing += (1.0 - frac)
            else:
                oar_over_max = max(oar_over_max, frac - 1.0)
        if dose > limit:
            oar_ok = False

        if use_baseline and organ in baseline_oar_doses:
            base = float(baseline_oar_doses.get(organ, 0.0))
            delta = float(dose) - base
            if delta > 0.0:
                denom = max(limit, 1e-6)
                oar_penalty += weight * ((delta / denom) ** 2)
            else:
                denom = max(limit, 1e-6)
                oar_sparing += (-delta) / denom
        else:
            if dose > limit:
                over = float(dose - limit)
                oar_penalty += weight * ((over / limit) ** 2)


   


    # -----------------------------
    # 4) STOP ACTION LOGIC
    # -----------------------------
    stop_penalty = 0.0
    coverage_frac = hrctv_d90 / max(target_d90, 1e-6)
    in_d90_band = (hrctv_d90 >= d90_low) and (hrctv_d90 <= d90_high)
    on_target_d90 = hrctv_d90 >= d90_low
    on_target_d98 = hrctv_d98 >= (target_d98 - d98_tol)
    if stop:
        # Only reward stopping when D90 is in-band; otherwise penalize.
        if in_d90_band:
            # Reward stopping in-band; larger bonus if D98 and OARs are acceptable.
            stop_penalty = -80.0
            if on_target_d98 and oar_ok:
                stop_penalty = -150.0
        else:
            band_violation = float(d90_low - hrctv_d90) if hrctv_d90 < d90_low else float(hrctv_d90 - d90_high)
            stop_penalty = 10.0 + 0.2 * (band_violation ** 2)
        if len(needle_positions) == 0 and hrctv_d90 < 600.0:
            # Strongly discourage stopping with zero needles while underdosed.
            stop_penalty += 100.0
    else:
        # Penalize continuing when goals are met and OARs are safe.
        if in_d90_band and on_target_d98 and oar_ok:
            stop_penalty = 25.0
        elif on_target_d90 and on_target_d98 and oar_ok:
            stop_penalty = 10.0


    # -----------------------------
    # 5) Combine reward terms
    # -----------------------------
    coverage_frac = hrctv_d90 / max(target_d90, 1e-6)
    coverage_frac_clipped = np.clip(coverage_frac, 0.0, 2.0)
    # Continuous shaped D90 term (symmetric around target)
    d90_shaped = -((hrctv_d90 - target_d90) / sigma90) ** 2
    d90_shaped_weight = 4.0  # tune to balance vs other terms

    coverage_bonus = 0.25 * coverage_200 + 0.75 * coverage_400 + 2.0 * coverage_600
    on_target_scale = np.clip((hrctv_d90 - d90_low) / max(d90_high - d90_low, 1e-6), 0.0, 1.0)
    _ = delta_d90  # unused in lexicographic reward

    # Lexicographic optimization:
    # 1) Enforce D90 band [600,700] as a hard constraint (quadratic penalty outside).
    # 2) Inside the band, minimize OAR dose (relative to baseline).
    band_low = d90_low
    band_high = d90_high
    if hrctv_d90 < band_low:
        band_violation = float(band_low - hrctv_d90)
    elif hrctv_d90 > band_high:
        band_violation = float(hrctv_d90 - band_high)
    else:
        band_violation = 0.0

    if band_violation > 0.0:
        if hrctv_d90 > band_high:
            band_weight = 0.6
            overshoot_weight = 0.10
        else:
            band_weight = 0.4
            overshoot_weight = 0.0
        reward = -band_weight * (band_violation ** 2)
        if overshoot_weight > 0.0:
            reward -= overshoot_weight * (band_violation ** 2)
        # Keep OAR penalty near-zero outside the D90 band.
        reward -= 0.0 * oar_penalty
        reward -= stop_penalty + penalty + zero_penalty + d90_overshoot_penalty
        return float(reward)

    # In-band: focus on OAR minimization.
    oar_weight_inband = 60.0
    oar_sparing_weight = 5.0
    dwell_penalty_weight = 0.02
    total_dwell = float(total_dwell_time or 0.0)
    reward = (
        - oar_weight_inband * oar_penalty
        + oar_sparing_weight * oar_sparing
        - dwell_penalty_weight * total_dwell
        - stop_penalty
        - penalty
        - zero_penalty
    )
    return float(reward)


    return float(reward)



def compute_tandem_reward(
    hrctv_d90,
    hrctv_d98,
    oar_doses,
    rx_total=700.0,
    tandem_fraction_target=455.0 / 700.0,
    OAR_limits=None,
    hrctv_mean=0.0,
    hrctv_coverages=None,
    vagina_penalty_multiplier=3.0,
):
    """
    Reward for tandem-only dwell time optimization.

    Tandem is intended to produce a higher HRCTV coverage target by default.
    - hrctv_d90: HRCTV D90 in cGy
    - oar_doses: dict with D2cc for each organ
    - rx_total: total prescription dose (cGy)
    - tandem_fraction_target: fraction of prescription expected from tandem
    """

    if OAR_limits is None:
        OAR_limits = OAR_CONSTRAINTS

    target_dose = tandem_fraction_target * rx_total
    target_d98 = 0.9 * target_dose
    target_overshoot = min(rx_total, target_dose * 1.1)  # discourage exceeding target by ~10%

    hrctv_term = np.clip(hrctv_d90 / max(target_dose, 1e-6), 0, 2.0)
    d98_term = np.clip(hrctv_d98 / max(target_d98, 1e-6), 0, 2.0)

    oar_penalty = 0.0
    for organ, dose in oar_doses.items():
        limit = OAR_limits.get(organ, 200.0)
        if dose > limit:
            over_frac = (dose - limit) / limit
            weight = OAR_PENALTY_WEIGHTS.get(organ, 1.0)
            if organ == "Vagina":
                weight *= float(vagina_penalty_multiplier)
            # Quadratic penalty scaled to discourage exceeding limits
            oar_penalty += weight * 10.0 * (over_frac ** 2)

    mean_term = np.clip(hrctv_mean / max(target_dose, 1e-6), 0, 2.0)

    partial_term = np.clip(hrctv_mean / max(target_dose * 0.5, 1e-6), 0, 2.0)

    coverage_term = 0.0
    if hrctv_coverages is not None and len(hrctv_coverages) > 0:
        # Expect coverage tuple (e.g., >=400 cGy, >=600 cGy)
        coverage_term = 1.5 * hrctv_coverages[0]
        if len(hrctv_coverages) > 1:
            coverage_term += 4.0 * hrctv_coverages[1]

    # Penalize overshoot beyond ~70% Rx to keep tandem modest
    overshoot_pen = 0.0
    if hrctv_d90 > target_overshoot:
        over_frac = (hrctv_d90 - target_overshoot) / target_overshoot
        overshoot_pen = 10.0 * (over_frac ** 2)

    d90_cap_penalty = 0.0
    if hrctv_d90 > D90_MAX_CGY:
        over_cgy = float(hrctv_d90 - D90_MAX_CGY)
        d90_cap_penalty = D90_OVERSHOOT_PENALTY * (over_cgy ** 2)

    reward = (
        + 6.0 * hrctv_term    # emphasize D90 more strongly
        + 1.5 * d98_term      # encourage D98 target
        + 15.0 * mean_term     # reward broader coverage
        + 8.0 * partial_term  # immediate reward for any HRCTV dose
        + coverage_term       # direct coverage incentive
        - 4.0 * oar_penalty   # strong OAR protection
        - overshoot_pen       # keep tandem under ~70% Rx
        - d90_cap_penalty     # enforce absolute D90 ceiling
    )

    return float(reward)
