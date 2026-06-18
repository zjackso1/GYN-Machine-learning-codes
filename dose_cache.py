import numpy as np
from numba import njit

from env.reward_logic import TRAINING_DOSE_SCALE


def build_path_kernel_cache(paths, structure_shape, tg43_kernel):
    """
    Precompute flattened voxel indices and kernel values for each dwell position
    along every path. Returns a list (per path) of lists (per dwell) containing
    (indices, values) arrays.
    """
    nz, ny, nx = structure_shape
    stride_z = ny * nx
    stride_y = nx

    cache = []
    offsets = tg43_kernel["offsets"]

    for path in paths:
        dwell_entries = []
        for pos in path:
            z = int(np.clip(round(pos[0]), 0, nz - 1))
            y = int(np.clip(round(pos[1]), 0, ny - 1))
            x = int(np.clip(round(pos[2]), 0, nx - 1))

            indices = []
            values = []
            for dz, dy, dx, kernel_value in offsets:
                zi = z + dz
                yi = y + dy
                xi = x + dx

                if zi < 0 or yi < 0 or xi < 0 or zi >= nz or yi >= ny or xi >= nx:
                    continue

                flat_idx = zi * stride_z + yi * stride_y + xi
                indices.append(flat_idx)
                values.append(kernel_value * TRAINING_DOSE_SCALE)

            dwell_entries.append(
                (
                    np.asarray(indices, dtype=np.int64),
                    np.asarray(values, dtype=np.float32),
                )
            )
        cache.append(dwell_entries)

    return cache


@njit(cache=True, nogil=True)
def scatter_add_dose(flat_dose, contributions_idx, contributions_val, weights):
    total_hits = 0
    for dwell_idx in range(weights.shape[0]):
        weight = weights[dwell_idx]
        if weight <= 0:
            continue
        idx_arr = contributions_idx[dwell_idx]
        val_arr = contributions_val[dwell_idx]
        if idx_arr.size == 0:
            continue
        total_hits += 1
        for n in range(idx_arr.size):
            flat_dose[idx_arr[n]] += weight * val_arr[n]
    return total_hits
