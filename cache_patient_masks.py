#!/usr/bin/env python3
"""Cache structure masks + metadata and write a patient manifest."""
    #python scripts/cache_patient_masks.py \
  #--data-root "/Users/gmoney/Desktop/RLResearch/BrachyRL/data" \
  #--cache-root "/Users/gmoney/Desktop/RLResearch/BrachyRL/data/cache" \
  #--manifest-path "/Users/gmoney/Desktop/RLResearch/BrachyRL/data/patient_manifest.json"

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Iterable, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.structure_utils import generate_structure_mask, save_structure_cache


def _parse_ids(spec: str) -> List[int]:
    ids: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(list(range(int(lo), int(hi) + 1)))
        else:
            ids.append(int(part))
    return ids


def _find_rtstruct(struct_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(struct_dir, "RTSTRUCT*.dcm")))
    if candidates:
        return candidates[0]
    candidates = sorted(glob.glob(os.path.join(struct_dir, "RS*.dcm")))
    if candidates:
        return candidates[0]
    all_dcms = sorted(glob.glob(os.path.join(struct_dir, "*.dcm")))
    if not all_dcms:
        raise FileNotFoundError(f"No DICOM files found in {struct_dir}")
    try:
        import pydicom
    except ImportError as exc:
        raise FileNotFoundError(
            f"No RTSTRUCT*.dcm found in {struct_dir} and pydicom is unavailable to probe files."
        ) from exc
    for path in all_dcms:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")).upper() == "RTSTRUCT":
            return path
    if len(all_dcms) == 1:
        return all_dcms[0]
    raise FileNotFoundError(
        f"No RTSTRUCT DICOM found in {struct_dir} (checked RTSTRUCT*, RS*, Modality==RTSTRUCT)."
    )


def _build_entries(
    data_root: str,
    patient_ids: Iterable[int],
    structures: List[str],
    cache_root: str,
    split: str,
    overwrite: bool,
) -> List[dict]:
    entries = []
    for pid in patient_ids:
        patient_name = f"Pt{pid} Fx1"
        patient_dir = os.path.join(data_root, patient_name)
        ct_series = os.path.join(patient_dir, "CT_Slices")
        rtstruct_dir = os.path.join(patient_dir, "STRUCT")
        rtstruct = _find_rtstruct(rtstruct_dir)

        cache_dir = os.path.join(cache_root, patient_name.replace(" ", "_"))
        meta = {
            "patient_id": patient_name,
            "rtstruct": rtstruct,
            "ct_series": ct_series,
            "structures": structures,
            "split": split,
        }

        if not overwrite and os.path.exists(os.path.join(cache_dir, "structure_mask.npy")):
            print(f"[SKIP] Cache exists for {patient_name}: {cache_dir}")
        else:
            print(f"[CACHE] Building {patient_name}")
            structure_mask, label_mapping, ct_spacing, ct_origin = generate_structure_mask(
                rtstruct, ct_series, structures
            )
            save_structure_cache(
                cache_dir,
                structure_mask,
                label_mapping,
                ct_spacing,
                ct_origin,
                meta=meta,
            )

        cache_baseline = os.path.join(cache_dir, "tandem_dose_map.npy")
        patient_baseline = os.path.join(patient_dir, "tandem_dose_map.npy")
        baseline_path = None
        if os.path.exists(cache_baseline):
            baseline_path = cache_baseline
        elif os.path.exists(patient_baseline):
            os.makedirs(cache_dir, exist_ok=True)
            if overwrite or not os.path.exists(cache_baseline):
                shutil.copy2(patient_baseline, cache_baseline)
                print(f"[CACHE] Copied tandem baseline -> {cache_baseline}")
            baseline_path = cache_baseline if os.path.exists(cache_baseline) else patient_baseline
        entry = {
            "patient_id": patient_name,
            "rtstruct": rtstruct,
            "ct_series": ct_series,
            "cache_dir": cache_dir,
            "split": split,
        }
        if baseline_path:
            entry["tandem_dose_map"] = baseline_path
        entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache structure masks and write a patient manifest."
    )
    parser.add_argument("--data-root", default="data", help="Root folder with patient directories.")
    parser.add_argument("--cache-root", default="data/cache", help="Output cache root.")
    parser.add_argument("--manifest-path", default="data/patient_manifest.json", help="Output manifest JSON.")
    parser.add_argument("--train-ids", default="1-7", help="Train patient IDs (e.g., 1-7,9).")
    parser.add_argument("--eval-ids", default="8-10", help="Eval patient IDs (e.g., 8-10).")
    parser.add_argument(
        "--structures",
        nargs="+",
        default=["HRCTV", "Rectum", "Bladder", "Sigmoid", "Bowel", "Vagina"],
        help="ROI names to rasterize into the structure mask.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caches.")
    args = parser.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)
    data_root = os.path.abspath(args.data_root)
    cache_root = os.path.abspath(args.cache_root)

    train_ids = _parse_ids(args.train_ids)
    eval_ids = _parse_ids(args.eval_ids)

    manifest = {
        "data_root": data_root,
        "cache_root": cache_root,
        "structures": args.structures,
        "patients": [],
    }
    manifest["patients"].extend(
        _build_entries(
            data_root,
            train_ids,
            args.structures,
            cache_root,
            split="train",
            overwrite=args.overwrite,
        )
    )
    manifest["patients"].extend(
        _build_entries(
            data_root,
            eval_ids,
            args.structures,
            cache_root,
            split="eval",
            overwrite=args.overwrite,
        )
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_path)), exist_ok=True)
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[DONE] Wrote manifest: {args.manifest_path}")


if __name__ == "__main__":
    main()
