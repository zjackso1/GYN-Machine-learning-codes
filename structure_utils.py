import json
import os
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder


def load_ct_series(dicom_folder):
    """
    Load DICOM CT series as a NumPy array.
    Returns:
        array  : np.ndarray (Z, Y, X)
        spacing: (sx, sy, sz) in mm
        origin : (x0, y0, z0) in mm
        image  : SimpleITK.Image
    """
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_IDs:
        raise FileNotFoundError(f"No DICOM series found in {dicom_folder}")
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_IDs[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    spacing = image.GetSpacing()           # (sx, sy, sz)
    origin = image.GetOrigin()
    return array, spacing, origin, image


def generate_structure_mask(rtstruct_path, ct_series_path, structures):
    """
    Create a 3D structure mask aligned to the CT series.
    Each structure gets a unique integer label starting from 1.

    Returns:
        structure_mask: np.ndarray (Z,Y,X)
        label_mapping : dict {structure_name: label}
    """
    ct_array, ct_spacing, ct_origin, ct_image = load_ct_series(ct_series_path)
    print(f"[INFO] CT loaded: shape={ct_array.shape}, spacing={ct_spacing}, origin={ct_origin}")

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=ct_series_path,
        rt_struct_path=rtstruct_path
    )

    label_mapping = {name: i + 1 for i, name in enumerate(structures)}
    structure_mask = np.zeros(ct_array.shape, dtype=np.int32)  # (Z,Y,X)

    for name in structures:
        if name not in rtstruct.get_roi_names():
            print(f"[WARN] '{name}' not found in RTSTRUCT.")
            continue

        mask = rtstruct.get_roi_mask_by_name(name)  # (X,Y,Z)
        print(f"[DEBUG] {name} mask shape (from RTSTRUCT): {mask.shape}, voxels={mask.sum()}")

        mask = mask.transpose(2, 1, 0)

        if mask.shape != ct_array.shape:
            print(f"[INFO] Resampling {name} mask to CT shape {ct_array.shape}")
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_sitk.SetSpacing(ct_spacing)
            mask_sitk.SetOrigin(ct_origin)
            mask_sitk = sitk.Resample(
                mask_sitk,
                ct_image,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8
            )
            mask = sitk.GetArrayFromImage(mask_sitk)

        structure_mask[mask > 0] = label_mapping[name]

    print(f"[INFO] Structure mask built: {structure_mask.shape}")
    for name, label in label_mapping.items():
        count = np.sum(structure_mask == label)
        print(f"  {name}: voxels={count}")

    return structure_mask, label_mapping, ct_spacing, ct_origin


def save_structure_cache(cache_dir,
                         structure_mask,
                         label_mapping,
                         ct_spacing,
                         ct_origin,
                         meta=None):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "structure_mask.npy"), structure_mask)
    np.save(os.path.join(cache_dir, "ct_spacing.npy"), np.asarray(ct_spacing, dtype=np.float32))
    np.save(os.path.join(cache_dir, "ct_origin.npy"), np.asarray(ct_origin, dtype=np.float32))
    with open(os.path.join(cache_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, sort_keys=True)
    if meta is not None:
        with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)


def load_structure_cache(cache_dir):
    mask_path = os.path.join(cache_dir, "structure_mask.npy")
    spacing_path = os.path.join(cache_dir, "ct_spacing.npy")
    origin_path = os.path.join(cache_dir, "ct_origin.npy")
    labels_path = os.path.join(cache_dir, "label_mapping.json")

    if not (os.path.exists(mask_path) and os.path.exists(spacing_path) and os.path.exists(origin_path)):
        raise FileNotFoundError(f"Missing cache files in {cache_dir}")
    structure_mask = np.load(mask_path)
    ct_spacing = tuple(np.load(spacing_path).tolist())
    ct_origin = tuple(np.load(origin_path).tolist())
    with open(labels_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    return structure_mask, label_mapping, ct_spacing, ct_origin
