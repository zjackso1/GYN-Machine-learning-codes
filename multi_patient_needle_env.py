from __future__ import annotations

import json
import os
import hashlib
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from env.anatomical_lib import build_bent_needle_library
from env.rt_brachy_env import BrachyRL_TG43
from env.structure_utils import generate_structure_mask, load_structure_cache
from env.tandem_geometry import build_tandem_angle_library
from env.rtplan_baseline import compute_rtplan_baseline_dose, find_rtplan, load_rtplan_paths

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MultiPatientNeedleEnv(gym.Env):
    """Sample a cached patient on reset and delegate to BrachyRL_TG43."""

    def __init__(
        self,
        patients: List[Dict[str, Any]],
        structures: List[str],
        tandem_length_mm: float = 70.0,
        tandem_step_mm: float = 5.0,
        depth_cm: float = 2.0,
        num_needles: int = 30,
        curve_points: int = 80,
        rng_seed: int = 42,
        slice_thickness_vox: float = 1.5,
        min_entry_sep_mm: Optional[float] = None,
        dwell_step_mm: float = 5.0,
        library_min_path_separation_mm: Optional[float] = None,
        entry_radius_mm: float = 20.0,
        entry_angle_limit_deg: float = 45.0,
        fixed_max_path_points: Optional[int] = None,
        require_baseline: bool = True,
        baseline_filename: str = "tandem_dose_map.npy",
        rng_seed_env: int = 0,
        **brachy_env_kwargs,
    ):
        if not patients:
            raise ValueError("No patients provided to MultiPatientNeedleEnv.")

        self.patients = list(patients)
        self.structures = list(structures)
        self.tandem_length_mm = float(tandem_length_mm)
        self.tandem_step_mm = float(tandem_step_mm)
        self.depth_cm = float(depth_cm)
        self.num_needles = int(num_needles)
        self.curve_points = int(curve_points)
        self.rng_seed = int(rng_seed)
        self.slice_thickness_vox = float(slice_thickness_vox)
        self.min_entry_sep_mm = min_entry_sep_mm
        self.dwell_step_mm = float(dwell_step_mm)
        self.library_min_path_separation_mm = library_min_path_separation_mm
        self.entry_radius_mm = float(entry_radius_mm)
        self.entry_angle_limit_deg = float(entry_angle_limit_deg)
        self.fixed_max_path_points = fixed_max_path_points
        self.require_baseline = bool(require_baseline)
        self.baseline_filename = str(baseline_filename)
        self.brachy_env_kwargs = dict(brachy_env_kwargs)

        self._rng = np.random.default_rng(rng_seed_env)
        self._cache: Dict[str, Dict[str, Any]] = {}

        if self.fixed_max_path_points is None:
            max_points = 0
            for entry in self.patients:
                payload = self._load_patient(entry)
                max_points = max(max_points, payload["max_path_points"])
            self.fixed_max_path_points = max_points

        self._env = self._build_env(self.patients[0])
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def _safe_name(self, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in str(name))
        return safe.strip("_") or "patient"

    def _hash_path(self, path_like) -> Optional[str]:
        arr = self._coerce_path(path_like)
        if arr is None or arr.size == 0:
            return None
        data = np.asarray(arr, dtype=np.float32).tobytes()
        return hashlib.sha1(data).hexdigest()

    def _anatomical_cache_key(
        self,
        structure_mask: np.ndarray,
        label_mapping: Dict[str, Any],
        axis_hint,
        os_vox_seed,
    ) -> dict:
        payload = {
            "version": 3,
            "structure_shape": list(structure_mask.shape),
            "label_mapping": dict(sorted(label_mapping.items())),
            "depth_cm": float(self.depth_cm),
            "num_needles": int(self.num_needles),
            "curve_points": int(self.curve_points),
            "rng_seed": int(self.rng_seed),
            "slice_thickness_vox": float(self.slice_thickness_vox),
            "min_entry_sep_mm": None if self.min_entry_sep_mm is None else float(self.min_entry_sep_mm),
            "dwell_step_mm": float(self.dwell_step_mm),
            "library_min_path_separation_mm": (
                None if self.library_min_path_separation_mm is None else float(self.library_min_path_separation_mm)
            ),
            "entry_radius_mm": float(self.entry_radius_mm),
            "entry_angle_limit_deg": float(self.entry_angle_limit_deg),
            "allow_vagina_path": False,
            "axis_hint_hash": self._hash_path(axis_hint),
            "os_vox_seed": None if os_vox_seed is None else [float(v) for v in np.asarray(os_vox_seed).reshape(-1)],
        }
        raw = json.dumps(payload, sort_keys=True)
        return {"cache_key": hashlib.sha1(raw.encode("utf-8")).hexdigest(), "params": payload}

    def _coerce_path(self, path_like) -> Optional[np.ndarray]:
        if path_like is None:
            return None
        if isinstance(path_like, dict):
            path_like = path_like.get("path_vox")
        if path_like is None:
            return None
        if isinstance(path_like, str):
            if not os.path.exists(path_like):
                return None
            path_like = np.load(path_like, allow_pickle=True)
        arr = np.asarray(path_like, dtype=np.float32)
        if arr.ndim != 2:
            return None
        if arr.shape[1] == 3:
            return arr
        if arr.shape[0] == 3:
            return arr.T
        return None

    def _coerce_paths(self, paths_like) -> List[np.ndarray]:
        if paths_like is None:
            return []
        if isinstance(paths_like, str):
            if not os.path.exists(paths_like):
                return []
            paths_like = np.load(paths_like, allow_pickle=True)
        if isinstance(paths_like, np.ndarray) and paths_like.dtype == object:
            iterable = list(paths_like)
        elif isinstance(paths_like, (list, tuple)):
            iterable = list(paths_like)
        else:
            single = self._coerce_path(paths_like)
            return [single] if single is not None else []
        out: List[np.ndarray] = []
        for item in iterable:
            path = self._coerce_path(item)
            if path is not None and path.size:
                out.append(path)
        return out

    def _resolve_actual_applicators(self, entry: Dict[str, Any]) -> tuple[Optional[np.ndarray], List[np.ndarray]]:
        tandem_keys = [
            "actual_tandem_path",
            "tandem_path",
            "tandem_path_file",
            "tandem_paths_file",
            "rtplan_tandem_path",
        ]
        ovoid_keys = [
            "actual_ovoid_paths",
            "ovoid_paths",
            "ovoid_paths_file",
            "rtplan_ovoid_paths",
        ]
        tandem = None
        for key in tandem_keys:
            if key in entry and entry.get(key):
                tandem = self._coerce_path(entry.get(key))
                if tandem is not None:
                    break
        ovoids: List[np.ndarray] = []
        for key in ovoid_keys:
            if key in entry and entry.get(key):
                ovoids = self._coerce_paths(entry.get(key))
                if ovoids:
                    break
        cache_dir = entry.get("cache_dir")
        if tandem is None and cache_dir:
            for fname in ("rtplan_tandem_path.npy", "tandem_path.npy", "tandem_paths.npy"):
                candidate = os.path.join(cache_dir, fname)
                if os.path.exists(candidate):
                    tandem = self._coerce_path(candidate)
                    if tandem is not None:
                        break
        if not ovoids and cache_dir:
            for fname in ("rtplan_ovoid_paths.npy", "ovoid_paths.npy"):
                candidate = os.path.join(cache_dir, fname)
                if os.path.exists(candidate):
                    ovoids = self._coerce_paths(candidate)
                    if ovoids:
                        break
        return tandem, ovoids

    def _resolve_baseline_path(self, entry: Dict[str, Any]) -> Optional[str]:
        candidates = []
        explicit = entry.get("tandem_dose_map")
        if explicit:
            candidates.append(explicit)
        cache_dir = entry.get("cache_dir")
        if cache_dir:
            candidates.append(os.path.join(cache_dir, self.baseline_filename))
        ct_series = entry.get("ct_series")
        if ct_series:
            patient_dir = os.path.dirname(ct_series)
            candidates.append(os.path.join(patient_dir, self.baseline_filename))
        for path in candidates:
            if path and os.path.exists(path):
                return os.path.abspath(path)
        return None

    def _resolve_tandem_angle(self, entry: Dict[str, Any]) -> Optional[float]:
        raw = entry.get("tandem_angle_deg", entry.get("tandem_angle"))
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

        candidates = []
        for key in ("tandem_angle_file", "tandem_angle_path"):
            angle_path = entry.get(key)
            if angle_path:
                candidates.append(angle_path)

        baseline_path = self._resolve_baseline_path(entry)
        if baseline_path:
            candidates.append(os.path.join(os.path.dirname(baseline_path), "tandem_angle.npy"))

        cache_dir = entry.get("cache_dir")
        if cache_dir:
            candidates.append(os.path.join(cache_dir, "tandem_angle.npy"))

        ct_series = entry.get("ct_series")
        if ct_series:
            patient_dir = os.path.dirname(ct_series)
            candidates.append(os.path.join(patient_dir, "tandem_angle.npy"))

        patient_id = entry.get("patient_id")
        if patient_id:
            safe_id = self._safe_name(patient_id)
            candidates.append(
                os.path.join(REPO_ROOT, "runs", "tandem_per_patient", f"{safe_id}_angle.npy")
            )

        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                return float(np.load(path).ravel()[0])
            except Exception:
                continue
        return None

    def _load_patient(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        key = entry.get("patient_id", entry.get("ct_series", "unknown"))
        if key in self._cache:
            return self._cache[key]

        cache_dir = entry.get("cache_dir")
        if cache_dir:
            structure_mask, label_mapping, ct_spacing, ct_origin = load_structure_cache(cache_dir)
        else:
            rtstruct = entry.get("rtstruct")
            ct_series = entry.get("ct_series")
            structure_mask, label_mapping, ct_spacing, ct_origin = generate_structure_mask(
                rtstruct, ct_series, self.structures
            )

        voxel_spacing = (ct_spacing[2], ct_spacing[1], ct_spacing[0])
        actual_tandem, actual_ovoids = self._resolve_actual_applicators(entry)
        if actual_tandem is not None or actual_ovoids:
            angle_deg = None
            os_vox_seed = None
            tandem_paths = [actual_tandem] if actual_tandem is not None else []
            ovoid_paths = actual_ovoids
            if actual_tandem is None:
                print(f"[WARN] No actual tandem path for patient '{key}'; using ovoids only.")
            if not ovoid_paths:
                print(f"[WARN] No actual ovoid paths for patient '{key}'.")
        else:
            angle_deg = self._resolve_tandem_angle(entry)
            angle_options = [float(angle_deg)] if angle_deg is not None else [15.0]
            tandem_library, os_vox_seed, ovoid_paths = build_tandem_angle_library(
                structure_mask=structure_mask,
                label_mapping=label_mapping,
                voxel_spacing=voxel_spacing,
                angle_options_deg=angle_options,
                length_mm=self.tandem_length_mm,
                step_mm=self.tandem_step_mm,
                include_ovoids=True,
            )
            tandem_paths = []
            if tandem_library:
                tandem_paths = [entry["path_vox"] for entry in tandem_library]

        baseline_path = self._resolve_baseline_path(entry)
        base_dose_map = None
        rtplan_tandem_path = None
        rtplan_ovoid_paths: List[np.ndarray] = []
        rtplan_path = entry.get("rtplan") or find_rtplan(entry.get("rtstruct"))
        ct_series = entry.get("ct_series")
        if cache_dir:
            cached_tandem = os.path.join(cache_dir, "rtplan_tandem_path.npy")
            if os.path.exists(cached_tandem):
                rtplan_tandem_path = self._coerce_path(cached_tandem)
            cached_ovoid = os.path.join(cache_dir, "rtplan_ovoid_paths.npy")
            if os.path.exists(cached_ovoid):
                rtplan_ovoid_paths = self._coerce_paths(cached_ovoid)

        if baseline_path:
            base_dose_map = np.load(baseline_path).astype(np.float32)
            if base_dose_map.shape != structure_mask.shape:
                raise ValueError(f"Baseline dose map shape mismatch for patient '{key}'.")

        if rtplan_path and ct_series and (rtplan_tandem_path is None or not rtplan_ovoid_paths):
            try:
                rtplan_tandem_path, rtplan_ovoid_paths = load_rtplan_paths(
                    rtplan_path=rtplan_path,
                    ct_series=ct_series,
                    structure_mask=structure_mask,
                )
                if cache_dir and rtplan_tandem_path is not None:
                    np.save(os.path.join(cache_dir, "rtplan_tandem_path.npy"), rtplan_tandem_path)
                if cache_dir and rtplan_ovoid_paths:
                    np.save(
                        os.path.join(cache_dir, "rtplan_ovoid_paths.npy"),
                        np.array(rtplan_ovoid_paths, dtype=object),
                    )
            except Exception as exc:
                print(f"[WARN] Failed to load RTPLAN applicator paths for patient '{key}': {exc}")

        if base_dose_map is None and rtplan_path and ct_series:
            try:
                base_dose_map, rtplan_tandem_path, rtplan_ovoid_paths = compute_rtplan_baseline_dose(
                    rtplan_path=rtplan_path,
                    ct_series=ct_series,
                    structure_mask=structure_mask,
                    label_mapping=label_mapping,
                    voxel_spacing=voxel_spacing,
                    dose_model=self.brachy_env_kwargs.get("dose_model", "blend"),
                    air_kerma_strength=self.brachy_env_kwargs.get("air_kerma_strength"),
                )
                baseline_path = f"rtplan:{os.path.basename(rtplan_path)}"
                print(f"[INFO] Computed baseline dose from RTPLAN for patient '{key}'.")
                # Cache baseline dose map to avoid recomputation.
                save_path = entry.get("tandem_dose_map")
                if not save_path and cache_dir:
                    save_path = os.path.join(cache_dir, self.baseline_filename)
                if not save_path and ct_series:
                    patient_dir = os.path.dirname(ct_series)
                    save_path = os.path.join(patient_dir, self.baseline_filename)
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, base_dose_map.astype(np.float32))
                    baseline_path = save_path
                    print(f"[INFO] Saved RTPLAN baseline dose to {save_path}.")
            except Exception as exc:
                print(f"[WARN] Failed to compute RTPLAN baseline for patient '{key}': {exc}")

        if rtplan_tandem_path is not None:
            tandem_paths = [rtplan_tandem_path]
            angle_deg = None
        if rtplan_ovoid_paths:
            ovoid_paths = rtplan_ovoid_paths

        axis_hint = tandem_paths[0] if tandem_paths else None
        anatomical_library = None
        cache_dir = entry.get("cache_dir")
        if cache_dir:
            lib_path = os.path.join(cache_dir, "anatomical_library.npy")
            meta_path = os.path.join(cache_dir, "anatomical_library_meta.json")
            if os.path.exists(lib_path) and os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    expected = self._anatomical_cache_key(
                        structure_mask=structure_mask,
                        label_mapping=label_mapping,
                        axis_hint=axis_hint,
                        os_vox_seed=os_vox_seed,
                    )
                    if meta.get("cache_key") == expected["cache_key"]:
                        anatomical_library = np.load(lib_path, allow_pickle=True).tolist()
                        print(f"[INFO] Loaded anatomical library from cache for patient '{key}'.")
                except Exception as exc:
                    print(f"[WARN] Failed to load anatomical library cache for '{key}': {exc}")

        if anatomical_library is None:
            anatomical_library = build_bent_needle_library(
                structure_mask=structure_mask,
                label_mapping=label_mapping,
                voxel_spacing=voxel_spacing,
                depth_cm=self.depth_cm,
                num_needles=self.num_needles,
                curve_points=self.curve_points,
                rng_seed=self.rng_seed,
                slice_thickness_vox=self.slice_thickness_vox,
                min_entry_separation_mm=self.min_entry_sep_mm,
                dwell_step_mm=self.dwell_step_mm,
                min_path_separation_mm=self.library_min_path_separation_mm,
                os_vox=os_vox_seed,
                entry_radius_mm=self.entry_radius_mm,
                entry_angle_limit_deg=self.entry_angle_limit_deg,
                world_origin=ct_origin,
                allow_vagina_path=False,
                axis_hint_zyx=axis_hint,
            )
            if cache_dir:
                try:
                    meta = self._anatomical_cache_key(
                        structure_mask=structure_mask,
                        label_mapping=label_mapping,
                        axis_hint=axis_hint,
                        os_vox_seed=os_vox_seed,
                    )
                    np.save(lib_path, np.array(anatomical_library, dtype=object))
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2, sort_keys=True)
                    print(f"[INFO] Saved anatomical library cache for patient '{key}'.")
                except Exception as exc:
                    print(f"[WARN] Failed to save anatomical library cache for '{key}': {exc}")
        if self.fixed_max_path_points is not None:
            capped = []
            max_len = int(self.fixed_max_path_points)
            for needle in anatomical_library:
                path = list(needle.get("path_vox", []))
                if len(path) == 0:
                    continue
                if len(path) > max_len:
                    path = path[:max_len]
                min_idx = int(needle.get("min_dwell_idx", 0) or 0)
                max_idx = needle.get("max_dwell_idx", None)
                if max_idx is not None:
                    max_idx = int(max_idx)
                min_idx = min(min_idx, len(path) - 1)
                if max_idx is None:
                    max_idx = len(path) - 1
                else:
                    max_idx = min(max_idx, len(path) - 1)
                if max_idx < min_idx:
                    continue
                capped.append({
                    "path_vox": path,
                    "min_dwell_idx": min_idx,
                    "max_dwell_idx": max_idx,
                })
            anatomical_library = capped

        if self.require_baseline and base_dose_map is None:
            raise ValueError(
                f"Missing tandem baseline for patient '{key}'. "
                "Add 'tandem_dose_map' to the manifest/cache or provide an RTPLAN with dwell data."
            )

        if not anatomical_library:
            print(f"[WARN] Anatomical library empty for patient '{key}'.")
        payload = {
            "patient_id": key,
            "structure_mask": structure_mask,
            "label_mapping": label_mapping,
            "ct_spacing": ct_spacing,
            "ct_origin": ct_origin,
            "voxel_spacing": voxel_spacing,
            "tandem_angle_deg": angle_deg,
            "anatomical_library": anatomical_library,
            "max_path_points": max((len(p["path_vox"]) for p in anatomical_library), default=0),
            "base_dose_map": base_dose_map,
            "baseline_path": baseline_path,
            "tandem_paths": tandem_paths,
            "ovoid_paths": ovoid_paths,
        }
        self._cache[key] = payload
        return payload

    def _build_env(self, entry: Dict[str, Any]) -> BrachyRL_TG43:
        payload = self._load_patient(entry)
        voxel_spacing = payload["voxel_spacing"]
        kwargs = dict(self.brachy_env_kwargs)
        kwargs.update(
            structure_mask=payload["structure_mask"],
            anatomical_library=payload["anatomical_library"],
            base_dose_map=payload["base_dose_map"],
            voxel_spacing_mm=voxel_spacing,
            voxel_size_mm=float(np.mean(voxel_spacing)),
            label_mapping=payload["label_mapping"],
            fixed_max_path_points=self.fixed_max_path_points,
            avoid_paths=payload.get("tandem_paths"),
            tandem_paths=payload.get("tandem_paths"),
            ovoid_paths=payload.get("ovoid_paths"),
            tandem_path_idx=0 if payload.get("tandem_paths") else None,
        )
        env = BrachyRL_TG43(**kwargs)
        env.ct_origin = payload["ct_origin"]
        env.ct_spacing = payload["ct_spacing"]
        env.patient_id = payload["patient_id"]
        env.managed_baseline = True
        env.baseline_path = payload["baseline_path"]
        env.tandem_angle_deg = payload.get("tandem_angle_deg")
        return env

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        entry = self._rng.choice(self.patients)
        self._env = self._build_env(entry)
        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info)
        info["patient_id"] = entry.get("patient_id")
        info["baseline_path"] = getattr(self._env, "baseline_path", None)
        info["tandem_angle_deg"] = getattr(self._env, "tandem_angle_deg", None)
        return obs, info

    def step(self, action):
        return self._env.step(action)

    def render(self, mode="human"):
        return self._env.render(mode=mode)
