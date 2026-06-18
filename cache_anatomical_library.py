#!/usr/bin/env python3
"""
Precompute and cache anatomical needle libraries for patients in a manifest.
This triggers MultiPatientNeedleEnv._load_patient, which now saves:
  cache_dir/anatomical_library.npy
  cache_dir/anatomical_library_meta.json
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.multi_patient_env import load_patient_manifest
from env.multi_patient_needle_env import MultiPatientNeedleEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache anatomical needle libraries for a manifest.")
    parser.add_argument("--manifest", required=True, help="Path to patient manifest JSON.")
    parser.add_argument("--split", default=None, help="Optional split name (train/eval).")
    parser.add_argument("--min-entry-sep-mm", type=float, default=None)
    parser.add_argument("--min-path-sep-mm", type=float, default=None)
    parser.add_argument("--num-needles", type=int, default=30)
    parser.add_argument("--curve-points", type=int, default=80)
    parser.add_argument("--depth-cm", type=float, default=2.0)
    parser.add_argument("--dwell-step-mm", type=float, default=5.0)
    parser.add_argument("--slice-thickness-vox", type=float, default=1.5)
    parser.add_argument("--entry-radius-mm", type=float, default=20.0)
    parser.add_argument("--entry-angle-limit-deg", type=float, default=45.0)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()

    patients = load_patient_manifest(args.manifest, split=args.split)
    if not patients:
        raise SystemExit("No patients found for the requested split.")

    structures = None
    try:
        import json
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        structures = manifest.get("structures")
    except Exception:
        structures = None
    if not structures:
        structures = ["HRCTV", "Rectum", "Bladder", "Sigmoid", "Bowel", "Vagina"]

    env = MultiPatientNeedleEnv(
        patients=patients,
        structures=structures,
        depth_cm=args.depth_cm,
        num_needles=args.num_needles,
        curve_points=args.curve_points,
        rng_seed=args.rng_seed,
        slice_thickness_vox=args.slice_thickness_vox,
        min_entry_sep_mm=args.min_entry_sep_mm,
        dwell_step_mm=args.dwell_step_mm,
        library_min_path_separation_mm=args.min_path_sep_mm,
        entry_radius_mm=args.entry_radius_mm,
        entry_angle_limit_deg=args.entry_angle_limit_deg,
        fixed_max_path_points=1,  # avoid prepass; we only want caching
        require_baseline=False,
    )

    failures = 0
    for entry in patients:
        try:
            env._load_patient(entry)
        except Exception as exc:
            failures += 1
            pid = entry.get("patient_id", "unknown")
            print(f"[WARN] Failed to cache anatomical library for {pid}: {exc}")

    if failures:
        print(f"[DONE] Cached anatomical libraries with {failures} failures.")
    else:
        print("[DONE] Cached anatomical libraries.")


if __name__ == "__main__":
    main()
