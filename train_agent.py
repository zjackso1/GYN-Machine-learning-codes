import os
import sys
import argparse
import json
import csv
import numpy as np

 #PYTHONPATH="/Users/gmoney/Desktop/RLResearch/BrachyRL" \
    #python env/train_agent.py \
    #--patient-manifest "/Users/gmoney/Desktop/RLResearch/BrachyRL/data/patient_manifest.json" \
    #--patient-split train \
    #--eval-patient-split eval""

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from env.rt_brachy_env import BrachyRL_TG43
from env.anatomical_lib import build_bent_needle_library, visualize_bent_needles
from env.tandem_geometry import build_tandem_angle_library
from env.structure_utils import generate_structure_mask
from env.multi_patient_env import load_patient_manifest
from env.multi_patient_needle_env import MultiPatientNeedleEnv
from env.reward_logic import compute_dose_metrics, compute_hrctv_coverage, compute_reward, D90_MAX_CGY
from env.rtplan_baseline import compute_rtplan_baseline_dose, find_rtplan, load_rtplan_paths

DEFAULT_MAX_NEEDLES = 10
DEFAULT_SLICE_SIZE = 32
DEFAULT_SLICE_OFFSETS = (-1, 0, 1)
DEFAULT_NUM_MASK_CHANNELS = 6
DEFAULT_DWELL_DS_STRIDE = 3


def _infer_fixed_max_path_points_from_obs_len(
    obs_len: int,
    max_needles: int = DEFAULT_MAX_NEEDLES,
    slice_size: int = DEFAULT_SLICE_SIZE,
    slice_offsets=DEFAULT_SLICE_OFFSETS,
    num_mask_channels: int = DEFAULT_NUM_MASK_CHANNELS,
    dwell_ds_stride: int = DEFAULT_DWELL_DS_STRIDE,
) -> int | None:
    if obs_len <= 0:
        return None
    slice_dim = len(slice_offsets) * 3 * (1 + num_mask_channels) * slice_size * slice_size
    fixed = (12 + 3) + 5 + slice_dim
    dwell_ds_len = obs_len - fixed
    if dwell_ds_len <= 0:
        return None
    min_flat = (dwell_ds_len - 1) * dwell_ds_stride + 1
    max_flat = dwell_ds_len * dwell_ds_stride
    candidates = []
    for mpp in range(1, 512):
        flat_len = max_needles * mpp
        if min_flat <= flat_len <= max_flat:
            candidates.append(mpp)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda m: abs(max_flat - max_needles * m))


def _infer_fixed_max_path_points_from_model(model_path: str) -> int | None:
    try:
        model = PPO.load(model_path, env=None, device="cpu")
    except Exception:
        return None
    obs_space = getattr(model, "observation_space", None)
    if obs_space is None or not hasattr(obs_space, "shape") or not obs_space.shape:
        return None
    obs_len = int(np.prod(obs_space.shape))
    return _infer_fixed_max_path_points_from_obs_len(obs_len)


def _load_run_config(run_dir: str) -> dict | None:
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_run_config(run_dir: str, config: dict) -> None:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def save_dose_figure(dose_map, output_path, title="Dose Map"):
    import matplotlib.pyplot as plt

    mid_z = dose_map.shape[0] // 2
    mid_y = dose_map.shape[1] // 2
    mid_x = dose_map.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slices = [
        ("Axial", dose_map[mid_z, :, :]),
        ("Sagittal", dose_map[:, mid_y, :]),
        ("Coronal", dose_map[:, :, mid_x]),
    ]
    for ax, (label, data) in zip(axes, slices):
        im = ax.imshow(data, cmap="hot")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_isodose_overlay(dose_map, structure_mask, label_mapping, output_path, rx_cgy=700.0):
    import matplotlib.pyplot as plt

    mid_z = dose_map.shape[0] // 2
    mid_y = dose_map.shape[1] // 2
    mid_x = dose_map.shape[2] // 2

    slices = [
        ("Axial", dose_map[mid_z, :, :], structure_mask[mid_z, :, :]),
        ("Sagittal", dose_map[:, mid_y, :], structure_mask[:, mid_y, :]),
        ("Coronal", dose_map[:, :, mid_x], structure_mask[:, :, mid_x]),
    ]

    levels = [0.5 * rx_cgy, 0.7 * rx_cgy, rx_cgy]
    structure_colors = {
        "HRCTV": "red",
        "Rectum": "orange",
        "Bladder": "cyan",
        "Sigmoid": "magenta",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, dose_slice, mask_slice) in zip(axes, slices):
        im = ax.imshow(dose_slice, cmap="inferno")
        ax.contour(dose_slice, levels=levels, colors=["lime", "yellow", "white"], linewidths=1.0)
        for organ, color in structure_colors.items():
            label_value = label_mapping.get(organ)
            if label_value is None:
                continue
            organ_mask = (mask_slice == label_value)
            if np.any(organ_mask):
                ax.contour(organ_mask.astype(float), levels=[0.5], colors=[color], linewidths=0.8)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Needle + Tandem Isodose Overlay")
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Dose (cGy)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_dwells_csv(dwells, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["needle_idx", "dwell_idx", "dwell_time"])
        for needle_idx, dwell_vec in enumerate(dwells):
            arr = np.asarray(dwell_vec, dtype=float).reshape(-1)
            for dwell_idx, val in enumerate(arr):
                writer.writerow([needle_idx, dwell_idx, float(val)])


class VecNormalizeSyncCallback(BaseCallback):
    """Keep eval VecNormalize stats in sync with the training env."""

    def __init__(self, train_env, eval_env):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        sync_envs_normalization(self.train_env, self.eval_env)
        return True


class NeedleEvalLogger(BaseCallback):
    """Periodically run a deterministic rollout and log HRCTV metrics."""

    def __init__(self, env_factory, eval_freq=5000, max_eval_steps=200):
        super().__init__()
        self.eval_freq = eval_freq
        self.max_eval_steps = max_eval_steps
        self.eval_env = DummyVecEnv([env_factory()])
        self.eval_env = VecNormalize(self.eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        self.eval_env.training = False
        self.eval_env.norm_reward = False
        self.baseline_d90 = None
        self.baseline_oars = None

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True

        if isinstance(self.training_env, VecNormalize):
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.clip_obs = self.training_env.clip_obs

        # Ensure evaluation env uses the same tandem seed/base dose map
        try:
            train_brachy = unwrap_to_brachy_env(self.training_env)
            eval_brachy = unwrap_to_brachy_env(self.eval_env)
            managed = getattr(train_brachy, "managed_baseline", False) or getattr(eval_brachy, "managed_baseline", False)
            if not managed:
                if train_brachy.base_dose_map is not None:
                    eval_brachy.set_base_dose_map(train_brachy.base_dose_map.copy())
                else:
                    eval_brachy.clear_base_dose_map()
        except Exception:
            pass

        obs = self.eval_env.reset()
        baseline_info = unwrap_to_brachy_env(self.eval_env).last_reset_info
        if baseline_info:
            self.baseline_d90 = baseline_info.get("hrctv_d90", 0.0)
            self.baseline_oars = baseline_info.get("oar_doses", None)
        done = np.array([False])
        last_info = {}
        last_reward = 0.0
        steps = 0

        max_steps = self.max_eval_steps if self.max_eval_steps is not None else 200
        while (not done[0]) and (steps < max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_env.step(action)
            done = dones
            last_info = infos[0]
            last_reward = float(rewards[0])
            steps += 1

        if not done[0]:
            print(f"[NEEDLE EVAL] Reached max eval steps ({max_steps}); terminating rollout early.")

        hrctv_d90 = last_info.get("hrctv_d90", 0.0)
        hrctv_d98 = last_info.get("hrctv_d98", 0.0)
        needles = last_info.get("needles", 0)
        oars = last_info.get("oar_doses", {})
        print(
            f"[NEEDLE EVAL] t={self.num_timesteps} | steps={steps} | needles={needles} | "
            f"HRCTV D90={hrctv_d90:.2f} (baseline {self.baseline_d90:.2f}) D98={hrctv_d98:.2f} | reward={last_reward:.2f}"
        )
        cov_200 = last_info.get("coverage_200")
        cov_400 = last_info.get("coverage_400")
        cov_600 = last_info.get("coverage_600")
        if cov_200 is not None:
            print(f"    Coverage >=200/400/600 cGy: {cov_200:.2f} / {cov_400:.2f} / {cov_600:.2f}")
        if self.baseline_oars:
            baseline_summary = " | ".join(
                f"{name}={float(dose):.1f}" for name, dose in self.baseline_oars.items()
            )
            print(f"    Baseline OAR D2cc | {baseline_summary}")
        if oars:
            summary = " | ".join(f"{name}={dose:.1f}" for name, dose in oars.items())
            print(f"    OAR D2cc | {summary}")
        return True


def unwrap_to_brachy_env(vec_env):
    """Return the underlying BrachyRL_TG43 env from a wrapped VecEnv."""
    venv = getattr(vec_env, "venv", vec_env)
    env = getattr(venv, "envs", [venv])[0]
    while True:
        if hasattr(env, "env"):
            env = env.env
            continue
        if hasattr(env, "_env"):
            env = env._env
            continue
        break
    return env

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the straight-needle PPO agent")
    parser.add_argument(
        "--tandem-dose-map",
        type=str,
        default=None,
        help="Optional .npy file with a tandem dose map to seed the PPO env",
    )
    parser.add_argument(
        "--patient-manifest",
        type=str,
        default=None,
        help="Path to cached patient manifest JSON for multi-patient training.",
    )
    parser.add_argument(
        "--patient-split",
        type=str,
        default="train",
        help="Manifest split name for training (e.g., train).",
    )
    parser.add_argument(
        "--eval-patient-split",
        type=str,
        default="eval",
        help="Manifest split name for evaluation (e.g., eval).",
    )
    parser.add_argument(
        "--eval-rollouts",
        type=int,
        default=1,
        help="Number of deterministic eval rollouts to pick the best plan",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Base RNG seed for eval rollouts (rollout i uses eval_seed + i).",
    )
    parser.add_argument(
        "--enable-tandem-opt",
        action="store_true",
        help="Enable tandem/ovoid dwell optimization action (disabled by default for speed).",
    )
    parser.add_argument(
        "--visualize-tandem",
        action="store_true",
        help="Overlay a tandem path in the final visualization",
    )
    parser.add_argument(
        "--tandem-angles",
        nargs="+",
        type=float,
        default=[15, 30, 45],
        help="Angle options (deg) for building the tandem library",
    )
    parser.add_argument(
        "--tandem-angle-idx",
        type=int,
        default=0,
        help="Index into --tandem-angles for visualization",
    )
    parser.add_argument(
        "--tandem-angle-deg",
        type=float,
        default=None,
        help="Override: directly specify the tandem angle (deg) to visualize",
    )
    parser.add_argument(
        "--min-entry-sep-mm",
        type=float,
        default=5.0,
        help="Minimum separation (mm) between vaginal entry points when building the needle library",
    )
    parser.add_argument(
        "--min-path-sep-mm",
        type=float,
        default=None,
        help="Minimum distance (mm) between inserted needles (clamped to >= 2.0 mm)",
    )
    parser.add_argument(
        "--overlap-penalty",
        type=float,
        default=5.0,
        help="Penalty weight applied when needles violate the min path separation",
    )
    parser.add_argument(
        "--save-plan-viz",
        action="store_true",
        help="Save an offscreen PyVista visualization of the final plan",
    )
    parser.add_argument(
        "--max-dwell-time",
        type=float,
        default=30.0,
        help="Maximum dwell time (s) per needle position in the RL environment",
    )
    parser.add_argument(
        "--delta-max-seconds",
        type=float,
        default=5.0,
        help="Max per-step dwell delta (s) for incremental needle updates.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Maximum number of environment steps per episode (default=max_needles*max_path_points).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and only run evaluation/export using a saved model.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a saved PPO model (used with --export-only).",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=str,
        default=None,
        help="Path to VecNormalize stats (used with --export-only).",
    )
    parser.add_argument(
        "--fixed-max-path-points",
        type=int,
        default=None,
        help="Optional cap on max dwell points per needle (used to match old models).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name (used for output directory under runs/).",
    )
    args = parser.parse_args()

    # Auto-resolve fixed_max_path_points for export-only runs to avoid obs size mismatches.
    if args.export_only and args.fixed_max_path_points is None:
        model_path = args.model_path
        run_dir = None
        if model_path:
            run_dir = os.path.dirname(model_path)
        else:
            run_name = args.run_name or "ppo_brachy_anatomical_v2"
            run_dir = os.path.join("runs", run_name)
            candidate = os.path.join(run_dir, "final_model.zip")
            if os.path.exists(candidate):
                model_path = candidate
        auto_fixed = None
        if run_dir:
            cfg = _load_run_config(run_dir)
            if cfg and cfg.get("fixed_max_path_points"):
                auto_fixed = int(cfg["fixed_max_path_points"])
        if auto_fixed is None and model_path and os.path.exists(model_path):
            auto_fixed = _infer_fixed_max_path_points_from_model(model_path)
        if auto_fixed is not None:
            args.fixed_max_path_points = int(auto_fixed)
            print(f"[INFO] Auto fixed_max_path_points={auto_fixed}")
        else:
            print("[WARN] Could not infer fixed_max_path_points; proceeding without override.")

    use_manifest = bool(args.patient_manifest)
    eval_rollouts = max(3, int(args.eval_rollouts))
    if eval_rollouts != int(args.eval_rollouts):
        print(f"[INFO] eval_rollouts increased to {eval_rollouts} to save top-3 plans.")
    structures = ["HRCTV", "Rectum", "Bladder", "Sigmoid", "Bowel", "Vagina"]
    if use_manifest:
        with open(args.patient_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_structures = manifest_data.get("structures")
        if manifest_structures:
            structures = list(manifest_structures)

# ============================================================
# 2. PATHS & STRUCTURES
# ============================================================

    MAX_NEEDLES = DEFAULT_MAX_NEEDLES
    structure_mask = None
    label_mapping = None
    ct_spacing = None
    ct_origin = None
    voxel_spacing = None
    HRCTV_LABEL = None
    rtplan_tandem_path = None
    rtplan_ovoid_paths = None
    fixed_max_path_points = None

    if use_manifest:
        train_patients = load_patient_manifest(args.patient_manifest, split=args.patient_split)
        if not train_patients:
            raise ValueError(f"No patients found for split '{args.patient_split}'.")
        eval_patients = load_patient_manifest(args.patient_manifest, split=args.eval_patient_split)
        if not eval_patients:
            eval_patients = train_patients
        if args.tandem_dose_map:
            print("[WARN] --tandem-dose-map ignored when using --patient-manifest.")

        all_patients = train_patients + [p for p in eval_patients if p not in train_patients]
        if args.fixed_max_path_points is not None:
            fixed_max_path_points = int(args.fixed_max_path_points)
        else:
            fixed_env = MultiPatientNeedleEnv(
                patients=all_patients,
                structures=structures,
                tandem_length_mm=70.0,
                tandem_step_mm=5.0,
                depth_cm=2.0,
                num_needles=30,
                curve_points=80,
                rng_seed=42,
                slice_thickness_vox=1.5,
                min_entry_sep_mm=args.min_entry_sep_mm,
                dwell_step_mm=5.0,
                library_min_path_separation_mm=None,
                max_needles=MAX_NEEDLES,
                min_path_separation_mm=args.min_path_sep_mm,
                overlap_penalty=args.overlap_penalty,
                max_dwell_time=args.max_dwell_time,
                delta_max_seconds=args.delta_max_seconds,
                max_episode_steps=args.max_episode_steps,
                enable_tandem_opt=args.enable_tandem_opt,
                require_baseline=True,
            )
            fixed_max_path_points = fixed_env.fixed_max_path_points
            del fixed_env

        def make_brachy_env():
            return MultiPatientNeedleEnv(
                patients=train_patients,
                structures=structures,
                tandem_length_mm=70.0,
                tandem_step_mm=5.0,
                depth_cm=2.0,
                num_needles=30,
                curve_points=80,
                rng_seed=42,
                slice_thickness_vox=1.5,
                min_entry_sep_mm=args.min_entry_sep_mm,
                dwell_step_mm=5.0,
                library_min_path_separation_mm=None,
                fixed_max_path_points=fixed_max_path_points,
                max_needles=MAX_NEEDLES,
                min_path_separation_mm=args.min_path_sep_mm,
                overlap_penalty=args.overlap_penalty,
                max_dwell_time=args.max_dwell_time,
                delta_max_seconds=args.delta_max_seconds,
                max_episode_steps=args.max_episode_steps,
                enable_tandem_opt=args.enable_tandem_opt,
                require_baseline=True,
            )

        def make_eval_brachy_env():
            return MultiPatientNeedleEnv(
                patients=eval_patients,
                structures=structures,
                tandem_length_mm=70.0,
                tandem_step_mm=5.0,
                depth_cm=2.0,
                num_needles=30,
                curve_points=80,
                rng_seed=42,
                slice_thickness_vox=1.5,
                min_entry_sep_mm=args.min_entry_sep_mm,
                dwell_step_mm=5.0,
                library_min_path_separation_mm=None,
                fixed_max_path_points=fixed_max_path_points,
                max_needles=MAX_NEEDLES,
                min_path_separation_mm=args.min_path_sep_mm,
                overlap_penalty=args.overlap_penalty,
                max_dwell_time=args.max_dwell_time,
                delta_max_seconds=args.delta_max_seconds,
                max_episode_steps=args.max_episode_steps,
                enable_tandem_opt=args.enable_tandem_opt,
                require_baseline=True,
            )
    else:
        rtstruct_path = "/Users/gmoney/Desktop/RLResearch/BrachyRL/data/Pt1 Fx1/STRUCT/RTSTRUCTPT1.dcm"
        ct_series_path = "/Users/gmoney/Desktop/RLResearch/BrachyRL/data/Pt1 Fx1/CT_Slices"

        structure_mask, label_mapping, ct_spacing, ct_origin = generate_structure_mask(
            rtstruct_path, ct_series_path, structures
        )

        HRCTV_LABEL = label_mapping.get("HRCTV", 1)
        hrctv_voxels = np.argwhere(structure_mask == HRCTV_LABEL)
        print("HRCTV voxel count:", len(hrctv_voxels))

        tandem_dose_map = None
        tandem_path = None
        rtplan_tandem_path = None
        rtplan_ovoid_paths = None
        rtplan_path = find_rtplan(rtstruct_path)
        if args.tandem_dose_map:
            tandem_path = os.path.expanduser(args.tandem_dose_map)
        else:
            candidate = os.path.join(REPO_ROOT, "tandem_dose_map.npy")
            if os.path.exists(candidate):
                tandem_path = candidate
        if tandem_path:
            print(f"[INFO] Loading tandem dose map from {tandem_path}")
            tandem_dose_map = np.load(tandem_path).astype(np.float32)
            if tandem_dose_map.shape != structure_mask.shape:
                raise ValueError("Tandem dose map shape must match structure mask")
        # Convert (sx, sy, sz) → (dz, dy, dx) to match (Z,Y,X)
        sx, sy, sz = ct_spacing
        voxel_spacing = (sz, sy, sx)

        if rtplan_path:
            try:
                rtplan_tandem_path, rtplan_ovoid_paths = load_rtplan_paths(
                    rtplan_path=rtplan_path,
                    ct_series=ct_series_path,
                    structure_mask=structure_mask,
                )
            except Exception as exc:
                print(f"[WARN] Failed to load RTPLAN applicator paths: {exc}")

        if tandem_dose_map is None and rtplan_path:
            try:
                tandem_dose_map, rtplan_tandem_path, rtplan_ovoid_paths = compute_rtplan_baseline_dose(
                    rtplan_path=rtplan_path,
                    ct_series=ct_series_path,
                    structure_mask=structure_mask,
                    label_mapping=label_mapping,
                    voxel_spacing=voxel_spacing,
                    dose_model="blend",
                )
                print(f"[INFO] Computed baseline dose from RTPLAN: {rtplan_path}")
            except Exception as exc:
                print(f"[WARN] Failed to compute RTPLAN baseline: {exc}")


        # ============================================================
        # 3. BUILD ANATOMICAL STRAIGHT-NEEDLE LIBRARY
        # ============================================================

        print("[INFO] Building anatomical straight-needle library...")
        tandem_library, os_vox_seed, ovoid_paths = build_tandem_angle_library(
            structure_mask=structure_mask,
            label_mapping=label_mapping,
            voxel_spacing=voxel_spacing,
            angle_options_deg=[float(a) for a in args.tandem_angles],
            length_mm=70,
            step_mm=5.0,
            include_ovoids=True,
        )
        tandem_avoid_paths = None
        if tandem_library:
            tandem_angle_deg = None
            if tandem_path:
                angle_candidates = [
                    os.path.join(os.path.dirname(tandem_path), "tandem_angle.npy"),
                    os.path.join(REPO_ROOT, "tandem_angle.npy"),
                ]
                for angle_path in angle_candidates:
                    if os.path.exists(angle_path):
                        try:
                            tandem_angle_deg = float(np.load(angle_path).ravel()[0])
                            break
                        except Exception:
                            tandem_angle_deg = None
            if tandem_angle_deg is None and args.tandem_angle_deg is not None:
                tandem_angle_deg = float(args.tandem_angle_deg)
            if tandem_angle_deg is None:
                tandem_angle_deg = float(args.tandem_angles[0])
            angles = [entry["angle_deg"] for entry in tandem_library]
            chosen_idx = int(np.argmin([abs(tandem_angle_deg - ang) for ang in angles]))
            tandem_avoid_paths = [tandem_library[chosen_idx]["path_vox"]]

        env_tandem_paths = [entry["path_vox"] for entry in tandem_library] if tandem_library else []
        env_ovoid_paths = ovoid_paths
        env_tandem_path_idx = chosen_idx if tandem_library else None
        if rtplan_tandem_path is not None:
            env_tandem_paths = [rtplan_tandem_path]
            env_tandem_path_idx = 0
            tandem_avoid_paths = [rtplan_tandem_path]
        if rtplan_ovoid_paths:
            env_ovoid_paths = rtplan_ovoid_paths
        anatomical_library = build_bent_needle_library(
            structure_mask=structure_mask,
            label_mapping=label_mapping,
            voxel_spacing=voxel_spacing,
            depth_cm=2.0,
            num_needles=30,
            curve_points=80,
            rng_seed=42,
            slice_thickness_vox=1.5,
            min_entry_separation_mm=args.min_entry_sep_mm,
            dwell_step_mm=5.0,
            min_path_separation_mm=None,
            os_vox=os_vox_seed,
            entry_radius_mm=20.0,
            world_origin=ct_origin,
            allow_vagina_path=False,
            axis_hint_zyx=rtplan_tandem_path,
        )

        print(f"[INFO] Anatomical library size: {len(anatomical_library)}")


        # ============================================================
        # 4. CREATE ENVIRONMENT (NEW DISCRETE / STOP VERSION)
        # ============================================================

        def make_brachy_env():
            env = BrachyRL_TG43(
                structure_mask=structure_mask,
                max_needles=MAX_NEEDLES,
                voxel_size_mm=float(np.mean(voxel_spacing)),
                tg43_kernel=None,
                anatomical_library=anatomical_library,
                base_dose_map=tandem_dose_map,
                voxel_spacing_mm=voxel_spacing,
                min_path_separation_mm=args.min_path_sep_mm,
                avoid_paths=tandem_avoid_paths,
                overlap_penalty=args.overlap_penalty,
                max_dwell_time=args.max_dwell_time,
                delta_max_seconds=args.delta_max_seconds,
                max_episode_steps=args.max_episode_steps,
                label_mapping=label_mapping,
                tandem_paths=env_tandem_paths,
                ovoid_paths=env_ovoid_paths,
                tandem_path_idx=env_tandem_path_idx,
            )
            return env

        def make_eval_brachy_env():
            return make_brachy_env()

    # Optional sanity check on the raw environment
    check_env(make_brachy_env(), warn=True)

    def env_factory():
        def _init():
            return make_brachy_env()
        return _init

    def monitor_env_factory():
        def _init():
            return Monitor(make_eval_brachy_env())
        return _init


    # ============================================================
    # 5. PPO MODEL (DISCRETE ACTION SPACE)
    # ============================================================

    RUN_NAME = args.run_name or "ppo_brachy_anatomical_v2"
    SAVE_DIR = "./runs/" + RUN_NAME
    os.makedirs(SAVE_DIR, exist_ok=True)
    _write_run_config(
        SAVE_DIR,
        {
            "fixed_max_path_points": fixed_max_path_points,
            "max_needles": MAX_NEEDLES,
            "slice_size": DEFAULT_SLICE_SIZE,
            "slice_offsets": list(DEFAULT_SLICE_OFFSETS),
            "num_mask_channels": DEFAULT_NUM_MASK_CHANNELS,
            "dwell_ds_stride": DEFAULT_DWELL_DS_STRIDE,
            "structures": structures,
            "enable_tandem_opt": bool(args.enable_tandem_opt),
        },
    )
    vecnorm_path = os.path.join(SAVE_DIR, "vecnormalize.pkl")
    if args.vecnorm_path:
        vecnorm_path = args.vecnorm_path

    vec_env = DummyVecEnv([env_factory()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([monitor_env_factory()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(SAVE_DIR, "best_model"),
        log_path=os.path.join(SAVE_DIR, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    sync_callback = VecNormalizeSyncCallback(vec_env, eval_env)
    max_eval_steps = args.max_episode_steps if args.max_episode_steps is not None else 200
    d90_logger = NeedleEvalLogger(monitor_env_factory, eval_freq=5000, max_eval_steps=max_eval_steps)
    callback = CallbackList([sync_callback, eval_callback, d90_logger])

    if args.export_only:
        model_path = args.model_path or os.path.join(SAVE_DIR, "final_model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(vecnorm_path):
            raise FileNotFoundError(f"VecNormalize stats not found at {vecnorm_path}")
        model = PPO.load(model_path, env=vec_env)
        print(f"[INFO] Loaded model from {model_path}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            batch_size=64,
            n_steps=1024,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,
            clip_range=0.2,
            target_kl=None,
            tensorboard_log=SAVE_DIR,
        )

        # ============================================================
        # 6. TRAIN
        # ============================================================

        num_timesteps = 2000  # increase as you like
        print(f"[INFO] Starting PPO training for {num_timesteps} steps...")
        model.learn(total_timesteps=num_timesteps, callback=callback)
        model.save(os.path.join(SAVE_DIR, "final_model"))
        vec_env.save(vecnorm_path)
        print("✅ Model training complete and saved.")


    # ============================================================
    # 7. TEST TRAINED MODEL (DETERMINISTIC ROLLOUT)
    # ============================================================

    best_info = None
    best_paths = None
    best_dose_map = None
    best_rollout_idx = None
    best_structure_mask = None
    best_label_mapping = None
    best_ct_origin = None
    best_ct_spacing = None
    best_voxel_spacing = None
    best_tandem_paths = None
    best_ovoid_paths = None
    best_patient_id = None
    top_candidates = []

    rollout_env = DummyVecEnv([monitor_env_factory()])
    rollout_env = VecNormalize.load(vecnorm_path, rollout_env)
    rollout_env.training = False
    rollout_env.norm_reward = False

    for roll_idx in range(max(1, eval_rollouts)):
        print(f"\n[EVAL] Starting rollout {roll_idx + 1}/{eval_rollouts}")
        obs = rollout_env.reset()
        base_eval_env = unwrap_to_brachy_env(rollout_env)
        baseline_info = getattr(base_eval_env, "last_reset_info", {}) or {}
        baseline_d90 = baseline_info.get("hrctv_d90", 0.0)
        print(f"    [EVAL] Baseline HRCTV D90 (tandem + seed) = {baseline_d90:.2f} cGy")
        baseline_oars = baseline_info.get("oar_doses", {})
        if baseline_oars:
            baseline_summary = " | ".join(
                f"{name}={float(dose):.1f}" for name, dose in baseline_oars.items()
            )
            print(f"    [EVAL] Baseline OAR D2cc | {baseline_summary}")
        patient_id = getattr(base_eval_env, "patient_id", None)
        if patient_id:
            print(f"    [EVAL] Patient ID = {patient_id}")
        done = np.array([False])
        step_idx = 0
        last_info = {}
        last_reward = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = rollout_env.step(action)
            step_idx += 1

            last_reward = float(rewards[0])
            last_info = infos[0]
            done = dones

            print(
                f"Step {step_idx:2d} | Needles={last_info['needles']:2d} | "
                f"Action={action[0]} | HRCTV D90={last_info['hrctv_d90']:.2f} | Reward={last_reward:.3f}"
            )

            hrctv_label = base_eval_env.label_mapping.get("HRCTV", 1)
            hrctv_mask = (base_eval_env.structure_mask == hrctv_label)
            hrctv_max = (
                base_eval_env.dose_map[hrctv_mask].max()
                if np.any(hrctv_mask) else 0.0
            )
            print(f"    [EVAL] HRCTV max dose so far: {hrctv_max:.2f}")

        candidate = {
            "rollout_idx": roll_idx + 1,
            "patient_id": patient_id,
            "reward": float(last_reward),
            "hrctv_d90": float(last_info.get("hrctv_d90", 0.0)),
            "needles": int(last_info.get("needles", len(base_eval_env.needle_paths))),
            "oar_doses": dict(last_info.get("oar_doses", {})),
            "paths": [list(path) for path in base_eval_env.needle_paths],
            "dwells": [np.asarray(dw, dtype=np.float32) for dw in base_eval_env.needle_dwells],
            "dose_map": base_eval_env.dose_map.copy(),
            "structure_mask": base_eval_env.structure_mask,
            "label_mapping": dict(base_eval_env.label_mapping),
            "ct_origin": getattr(base_eval_env, "ct_origin", None),
            "ct_spacing": getattr(base_eval_env, "ct_spacing", None),
            "voxel_spacing": tuple(base_eval_env.voxel_spacing_mm.tolist()),
            "tandem_paths": [list(p) for p in getattr(base_eval_env, "tandem_paths", [])],
            "ovoid_paths": [list(p) for p in getattr(base_eval_env, "ovoid_paths", [])],
            "info": dict(last_info),
        }
        candidate["d90_over_cap"] = candidate["hrctv_d90"] > float(D90_MAX_CGY)
        top_candidates.append(candidate)

        if candidate["d90_over_cap"]:
            print(
                f"[WARN] Rollout {candidate['rollout_idx']} D90={candidate['hrctv_d90']:.2f} "
                f"exceeds cap {float(D90_MAX_CGY):.0f} cGy; excluding from best selection."
            )
        elif (
            best_info is None or
            candidate["hrctv_d90"] > best_info["hrctv_d90"] or
            candidate["reward"] > best_info.get("reward", -np.inf)
        ):
            best_info = dict(candidate["info"])
            best_info["reward"] = candidate["reward"]
            best_paths = candidate["paths"]
            best_dose_map = candidate["dose_map"]
            best_structure_mask = candidate["structure_mask"]
            best_label_mapping = candidate["label_mapping"]
            best_ct_origin = candidate["ct_origin"]
            best_ct_spacing = candidate["ct_spacing"]
            best_voxel_spacing = candidate["voxel_spacing"]
            best_patient_id = candidate["patient_id"]
            best_rollout_idx = candidate["rollout_idx"]
            best_tandem_paths = candidate.get("tandem_paths")
            best_ovoid_paths = candidate.get("ovoid_paths")

        print(
            f"[EVAL] Rollout {roll_idx + 1} complete: HRCTV D90={last_info['hrctv_d90']:.2f} cGy, "
            f"needles={last_info['needles']}"
        )

    if best_info is None:
        if not top_candidates:
            raise RuntimeError("Evaluation failed to produce any rollouts")
        print(
            f"[WARN] All rollouts exceeded D90 cap {float(D90_MAX_CGY):.0f} cGy; "
            "falling back to best reward."
        )
        fallback = max(top_candidates, key=lambda c: c["reward"])
        best_info = dict(fallback["info"])
        best_info["reward"] = fallback["reward"]
        best_paths = fallback["paths"]
        best_dose_map = fallback["dose_map"]
        best_structure_mask = fallback["structure_mask"]
        best_label_mapping = fallback["label_mapping"]
        best_ct_origin = fallback["ct_origin"]
        best_ct_spacing = fallback["ct_spacing"]
        best_voxel_spacing = fallback["voxel_spacing"]
        best_patient_id = fallback["patient_id"]
        best_rollout_idx = fallback["rollout_idx"]
        best_tandem_paths = fallback.get("tandem_paths")
        best_ovoid_paths = fallback.get("ovoid_paths")

    def _lexi_key(entry):
        d90 = float(entry.get("hrctv_d90", 0.0))
        in_band = 1 if (600.0 <= d90 <= 700.0) else 0
        band_dist = abs(d90 - 650.0)
        reward = float(entry.get("reward", -np.inf))
        return (in_band, -band_dist, reward)

    valid_candidates = [c for c in top_candidates if not c["d90_over_cap"]]
    if not valid_candidates:
        print(
            f"[WARN] No rollouts under D90 cap {float(D90_MAX_CGY):.0f} cGy; "
            "top-3 will include over-cap plans."
        )
        valid_candidates = top_candidates
    valid_candidates.sort(key=_lexi_key, reverse=True)
    top_plans = valid_candidates[:3]

    print(f"\n[INFO] Best evaluation rollout: #{best_rollout_idx}")
    if best_patient_id:
        print(f"  Patient ID: {best_patient_id}")
    print(f"  Needles used: {best_info['needles']}")
    print(f"  HRCTV D90: {best_info['hrctv_d90']:.2f} cGy")
    for organ, dose in best_info["oar_doses"].items():
        print(f"  {organ:<8s} D2cc: {dose:.2f} cGy")
    if top_plans:
        print("  Top-3 rollouts:")
        for rank, entry in enumerate(top_plans, start=1):
            pid = entry.get("patient_id") or "unknown"
            print(
                f"    #{rank}: rollout={entry['rollout_idx']} patient={pid} "
                f"needles={entry['needles']} D90={entry['hrctv_d90']:.2f} reward={entry['reward']:.2f}"
            )

    if best_structure_mask is None:
        best_structure_mask = structure_mask
    if best_label_mapping is None:
        best_label_mapping = label_mapping
    if best_voxel_spacing is None and voxel_spacing is not None:
        best_voxel_spacing = voxel_spacing
    if best_ct_spacing is None:
        best_ct_spacing = ct_spacing
    if best_ct_origin is None:
        best_ct_origin = ct_origin

    if best_dose_map is not None:
        np.save(os.path.join(SAVE_DIR, "needle_dose_map.npy"), best_dose_map)
        save_dose_figure(best_dose_map, os.path.join(SAVE_DIR, "needle_dose.png"), title="Needle Dose Map")
        save_isodose_overlay(
            best_dose_map,
            best_structure_mask,
            best_label_mapping,
            os.path.join(SAVE_DIR, "needle_isodose.png"),
        )
    if best_paths is not None and len(best_paths) > 0:
        np.save(os.path.join(SAVE_DIR, "best_paths.npy"), np.array(best_paths, dtype=object))

    def _to_jsonable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        return obj

    if top_plans:
        top_summary = []
        for rank, entry in enumerate(top_plans, start=1):
            prefix = f"top{rank}"
            dose_map = entry.get("dose_map")
            structure_mask_rank = entry.get("structure_mask")
            if structure_mask_rank is None:
                structure_mask_rank = best_structure_mask
            label_mapping_rank = entry.get("label_mapping")
            if label_mapping_rank is None:
                label_mapping_rank = best_label_mapping
            paths = entry.get("paths") or []
            dwells = entry.get("dwells") or []
            if dose_map is not None:
                np.save(os.path.join(SAVE_DIR, f"{prefix}_needle_dose_map.npy"), dose_map)
                save_dose_figure(
                    dose_map,
                    os.path.join(SAVE_DIR, f"{prefix}_needle_dose.png"),
                    title=f"Needle Dose Map ({prefix})",
                )
                save_isodose_overlay(
                    dose_map,
                    structure_mask_rank,
                    label_mapping_rank,
                    os.path.join(SAVE_DIR, f"{prefix}_needle_isodose.png"),
                )
            if paths:
                np.save(os.path.join(SAVE_DIR, f"{prefix}_paths.npy"), np.array(paths, dtype=object))
            if dwells:
                np.save(os.path.join(SAVE_DIR, f"{prefix}_needle_dwells.npy"), np.array(dwells, dtype=object))
                save_dwells_csv(dwells, os.path.join(SAVE_DIR, f"{prefix}_needle_dwells.csv"))
            top_summary.append({
                "rank": rank,
                "rollout_idx": entry.get("rollout_idx"),
                "patient_id": entry.get("patient_id"),
                "needles": entry.get("needles"),
                "hrctv_d90": entry.get("hrctv_d90"),
                "reward": entry.get("reward"),
                "oar_doses": entry.get("oar_doses"),
                "dose_map_file": f"{prefix}_needle_dose_map.npy",
                "paths_file": f"{prefix}_paths.npy" if paths else None,
                "dwells_file": f"{prefix}_needle_dwells.npy" if dwells else None,
                "dwells_csv": f"{prefix}_needle_dwells.csv" if dwells else None,
                "isodose_file": f"{prefix}_needle_isodose.png",
            })
        with open(os.path.join(SAVE_DIR, "top_eval_plans.json"), "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(top_summary), f, indent=2)

    if use_manifest:
        per_patient_dir = os.path.join(SAVE_DIR, "needle_per_patient")
        os.makedirs(per_patient_dir, exist_ok=True)

        def _safe_name(name: str) -> str:
            safe = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in str(name))
            return safe.strip("_") or "patient"

        def _run_eval_rollouts(entry, rollouts: int):
            results = []
            max_steps = args.max_episode_steps if args.max_episode_steps is not None else 200
            for roll_idx in range(max(1, int(rollouts))):
                rollout_seed = int(args.eval_seed) + int(roll_idx)
                eval_env = DummyVecEnv([
                    lambda entry=entry: MultiPatientNeedleEnv(
                        patients=[entry],
                        structures=structures,
                        tandem_length_mm=70.0,
                        tandem_step_mm=5.0,
                        depth_cm=2.0,
                        num_needles=30,
                        curve_points=80,
                        rng_seed=rollout_seed,
                        slice_thickness_vox=1.5,
                        min_entry_sep_mm=args.min_entry_sep_mm,
                        dwell_step_mm=5.0,
                        library_min_path_separation_mm=None,
                        fixed_max_path_points=fixed_max_path_points,
                        max_needles=MAX_NEEDLES,
                        min_path_separation_mm=args.min_path_sep_mm,
                        overlap_penalty=args.overlap_penalty,
                        max_dwell_time=args.max_dwell_time,
                        delta_max_seconds=args.delta_max_seconds,
                        max_episode_steps=args.max_episode_steps,
                        enable_tandem_opt=args.enable_tandem_opt,
                        require_baseline=True,
                        rng_seed_env=rollout_seed,
                    )
                ])
                eval_env = VecNormalize.load(vecnorm_path, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False

                obs = eval_env.reset()
                base_env = unwrap_to_brachy_env(eval_env)
                baseline_info = getattr(base_env, "last_reset_info", {}) or {}
                done = np.array([False])
                last_info = {}
                last_reward = 0.0
                steps = 0
                while (not done[0]) and (steps < max_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = eval_env.step(action)
                    done = dones
                    last_info = infos[0]
                    last_reward = float(rewards[0])
                    steps += 1

                # Compute a final "STOP" reward based on the terminal dose state
                (
                    hrctv_d90_final,
                    hrctv_d98_final,
                    hrctv_mean_final,
                    rectum_final,
                    bladder_final,
                    sigmoid_final,
                    bowel_final,
                    vagina_final,
                ) = compute_dose_metrics(
                    base_env.dose_map,
                    base_env.structure_mask,
                    voxel_spacing_mm=base_env.voxel_spacing_mm,
                )
                cov_200_final, cov_400_final, cov_600_final = compute_hrctv_coverage(
                    base_env.dose_map,
                    base_env.structure_mask,
                    thresholds=(200.0, 400.0, 600.0),
                )
                oar_final = {
                    "Rectum": rectum_final,
                    "Bladder": bladder_final,
                    "Sigmoid": sigmoid_final,
                    "Bowel": bowel_final,
                    "Vagina": vagina_final,
                }
                final_penalty = 0.0
                if hasattr(base_env, "_final_spacing_penalty"):
                    try:
                        final_penalty = float(base_env._final_spacing_penalty())
                    except Exception:
                        final_penalty = 0.0
                stop_reward = compute_reward(
                    hrctv_d90_final,
                    hrctv_d98_final,
                    hrctv_mean_final,
                    cov_200_final,
                    cov_400_final,
                    cov_600_final,
                    oar_final,
                    getattr(base_env, "baseline_oar_doses", None),
                    base_env.needle_positions,
                    delta_d90=0.0,
                    total_dwell_time=base_env._total_needle_dwell_time(),
                    penalty=final_penalty,
                    stop=True,
                )
                last_reward = float(stop_reward)

                results.append({
                    "rollout_idx": roll_idx + 1,
                    "last_info": dict(last_info),
                    "last_reward": float(last_reward),
                    "baseline_info": dict(baseline_info),
                    "steps": int(steps),
                    "dose_map": base_env.dose_map.copy(),
                    "structure_mask": base_env.structure_mask,
                    "label_mapping": dict(base_env.label_mapping),
                    "voxel_spacing_mm": getattr(base_env, "voxel_spacing_mm", None),
                    "paths": [list(path) for path in base_env.needle_paths],
                    "dwells": [np.asarray(dw, dtype=np.float32) for dw in base_env.needle_dwells],
                    "baseline_map": (
                        base_env.base_dose_map.copy()
                        if getattr(base_env, "base_dose_map", None) is not None
                        else None
                    ),
                })
            return results

        summary = []
        for entry in eval_patients:
            patient_id = entry.get("patient_id", "patient")
            safe_id = _safe_name(patient_id)
            results = _run_eval_rollouts(entry, eval_rollouts)
            if not results:
                raise RuntimeError(f"Evaluation failed for patient '{patient_id}'.")

            candidates = []
            for res in results:
                dose_map = res["dose_map"]
                structure_mask = res["structure_mask"]
                voxel_spacing_mm = res["voxel_spacing_mm"]
                (
                    hrctv_d90_calc,
                    hrctv_d98_calc,
                    hrctv_mean_calc,
                    rectum_calc,
                    bladder_calc,
                    sigmoid_calc,
                    bowel_calc,
                    vagina_calc,
                ) = compute_dose_metrics(
                    dose_map,
                    structure_mask,
                    voxel_spacing_mm=voxel_spacing_mm,
                )
                cov_200_calc, cov_400_calc, cov_600_calc = compute_hrctv_coverage(
                    dose_map,
                    structure_mask,
                    thresholds=(200.0, 400.0, 600.0),
                )
                oars_calc = {
                    "Rectum": rectum_calc,
                    "Bladder": bladder_calc,
                    "Sigmoid": sigmoid_calc,
                    "Bowel": bowel_calc,
                    "Vagina": vagina_calc,
                }
                last_info = res["last_info"]
                last_reward = res["last_reward"]
                baseline_info = res["baseline_info"]
                baseline_d90 = float(baseline_info.get("hrctv_d90", 0.0))
                baseline_d98 = float(baseline_info.get("hrctv_d98", 0.0))
                baseline_oars = {
                    k: float(v) for k, v in baseline_info.get("oar_doses", {}).items()
                }
                baseline_map = res.get("baseline_map")
                if baseline_map is not None:
                    (
                        base_d90,
                        base_d98,
                        _base_mean,
                        base_rectum,
                        base_bladder,
                        base_sigmoid,
                        base_bowel,
                        base_vagina,
                    ) = compute_dose_metrics(
                        baseline_map,
                        structure_mask,
                        voxel_spacing_mm=voxel_spacing_mm,
                    )
                    baseline_d90 = float(base_d90)
                    baseline_d98 = float(base_d98)
                    baseline_oars = {
                        "Rectum": float(base_rectum),
                        "Bladder": float(base_bladder),
                        "Sigmoid": float(base_sigmoid),
                        "Bowel": float(base_bowel),
                        "Vagina": float(base_vagina),
                    }

                candidates.append({
                    "rollout_idx": int(res["rollout_idx"]),
                    "steps": int(res["steps"]),
                    "needles": int(last_info.get("needles", len(res["paths"]))),
                    "hrctv_d90": float(hrctv_d90_calc),
                    "hrctv_d98": float(hrctv_d98_calc),
                    "hrctv_mean": float(hrctv_mean_calc),
                    "coverage_200": float(cov_200_calc),
                    "coverage_400": float(cov_400_calc),
                    "coverage_600": float(cov_600_calc),
                    "oar_doses": {k: float(v) for k, v in oars_calc.items()},
                    "baseline_d90": float(baseline_d90),
                    "baseline_d98": float(baseline_d98),
                    "baseline_oars": {k: float(v) for k, v in baseline_oars.items()},
                    "reward": float(last_reward),
                    "dose_map": dose_map,
                    "paths": res["paths"],
                    "dwells": res["dwells"],
                    "structure_mask": structure_mask,
                    "label_mapping": res["label_mapping"],
                })

            def _lexi_key(entry):
                d90 = float(entry.get("hrctv_d90", 0.0))
                in_band = 1 if (600.0 <= d90 <= 700.0) else 0
                band_dist = abs(d90 - 650.0)
                reward = float(entry.get("reward", -np.inf))
                return (in_band, -band_dist, reward)

            candidates.sort(key=_lexi_key, reverse=True)
            top_plans = candidates[:3]
            best = top_plans[0]

            # Save top-3 per patient
            per_patient_top = []
            for rank, entry in enumerate(top_plans, start=1):
                prefix = f"{safe_id}_top{rank}"
                dose_map = entry["dose_map"]
                np.save(os.path.join(per_patient_dir, f"{prefix}_dose_map.npy"), dose_map)
                save_dose_figure(
                    dose_map,
                    os.path.join(per_patient_dir, f"{prefix}_dose.png"),
                    title=f"Needle Dose Map ({patient_id}) {prefix}",
                )
                save_isodose_overlay(
                    dose_map,
                    entry["structure_mask"],
                    entry["label_mapping"],
                    os.path.join(per_patient_dir, f"{prefix}_isodose.png"),
                )
                if entry["paths"]:
                    np.save(os.path.join(per_patient_dir, f"{prefix}_paths.npy"), np.array(entry["paths"], dtype=object))
                if entry["dwells"]:
                    np.save(os.path.join(per_patient_dir, f"{prefix}_needle_dwells.npy"), np.array(entry["dwells"], dtype=object))
                    save_dwells_csv(entry["dwells"], os.path.join(per_patient_dir, f"{prefix}_needle_dwells.csv"))

                per_patient_top.append({
                    "rank": rank,
                    "rollout_idx": entry["rollout_idx"],
                    "needles": entry["needles"],
                    "hrctv_d90": entry["hrctv_d90"],
                    "reward": entry["reward"],
                    "oar_doses": entry["oar_doses"],
                    "dose_map_file": f"{prefix}_dose_map.npy",
                    "paths_file": f"{prefix}_paths.npy" if entry["paths"] else None,
                    "dwells_file": f"{prefix}_needle_dwells.npy" if entry["dwells"] else None,
                    "dwells_csv": f"{prefix}_needle_dwells.csv" if entry["dwells"] else None,
                    "isodose_file": f"{prefix}_isodose.png",
                })

            # Keep legacy single-plan filenames pointing to best plan
            np.save(os.path.join(per_patient_dir, f"{safe_id}_dose_map.npy"), best["dose_map"])
            np.save(os.path.join(per_patient_dir, f"{safe_id}_paths.npy"), np.array(best["paths"], dtype=object))
            np.save(os.path.join(per_patient_dir, f"{safe_id}_needle_dwells.npy"), np.array(best["dwells"], dtype=object))
            save_dwells_csv(best["dwells"], os.path.join(per_patient_dir, f"{safe_id}_needle_dwells.csv"))
            save_dose_figure(
                best["dose_map"],
                os.path.join(per_patient_dir, f"{safe_id}_dose.png"),
                title=f"Needle Dose Map ({patient_id})",
            )
            save_isodose_overlay(
                best["dose_map"],
                best["structure_mask"],
                best["label_mapping"],
                os.path.join(per_patient_dir, f"{safe_id}_isodose.png"),
            )

            summary.append(
                {
                    "patient_id": patient_id,
                    "steps": int(best["steps"]),
                    "needles": int(best["needles"]),
                    "hrctv_d90": float(best["hrctv_d90"]),
                    "hrctv_d98": float(best["hrctv_d98"]),
                    "hrctv_mean": float(best["hrctv_mean"]),
                    "coverage_200": float(best["coverage_200"]),
                    "coverage_400": float(best["coverage_400"]),
                    "coverage_600": float(best["coverage_600"]),
                    "oar_doses": {k: float(v) for k, v in best["oar_doses"].items()},
                    "baseline_d90": float(best["baseline_d90"]),
                    "baseline_d98": float(best["baseline_d98"]),
                    "baseline_oars": {k: float(v) for k, v in best["baseline_oars"].items()},
                    "reward": float(best["reward"]),
                    "top3": per_patient_top,
                }
            )

        if summary:
            with open(os.path.join(per_patient_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[INFO] Saved per-patient needle plans to {per_patient_dir}")

    # ============================================================
    # 8. VISUALIZE FINAL NEEDLES / SAVE PLAN SNAPSHOT
    # ============================================================

    viz_needed = args.visualize_tandem or args.save_plan_viz
    tandem_paths_for_viz = None
    ovoid_paths_for_viz = None
    if viz_needed:
        if best_structure_mask is None or best_label_mapping is None or best_voxel_spacing is None:
            print("[WARN] Skipping visualization; missing patient geometry.")
            viz_needed = False
        elif best_tandem_paths:
            tandem_paths_for_viz = best_tandem_paths
            ovoid_paths_for_viz = best_ovoid_paths if best_ovoid_paths else None
            print("[INFO] Using env tandem/ovoid paths for visualization.")
        else:
            angle_opts = [float(a) for a in args.tandem_angles]
            tandem_library, os_vox, _ = build_tandem_angle_library(
                structure_mask=best_structure_mask,
                label_mapping=best_label_mapping,
                voxel_spacing=best_voxel_spacing,
                angle_options_deg=angle_opts,
                length_mm=50,
                step_mm=5.0,
            )
            if args.tandem_angle_deg is not None:
                diffs = [abs(args.tandem_angle_deg - ang) for ang in angle_opts]
                chosen_idx = int(np.argmin(diffs))
            else:
                chosen_idx = int(np.clip(args.tandem_angle_idx, 0, len(angle_opts) - 1))
            tandem_entry = tandem_library[chosen_idx]
            tandem_paths_for_viz = [tandem_entry["path_vox"]]
            print(f"[INFO] Tandem angle for visualization: {tandem_entry['angle_deg']}°")
        if rtplan_tandem_path is not None and not best_tandem_paths:
            tandem_paths_for_viz = [rtplan_tandem_path]
            ovoid_paths_for_viz = rtplan_ovoid_paths if rtplan_ovoid_paths else None
            print("[INFO] Using RTPLAN tandem/ovoid paths for visualization.")

    final_library = [{"path_vox": path} for path in (best_paths or [])]
    affine = None
    if best_ct_spacing is not None and best_ct_origin is not None:
        sx, sy, sz = best_ct_spacing
        affine = np.eye(4, dtype=float)
        affine[0, 0] = sx
        affine[1, 1] = sy
        affine[2, 2] = sz
        affine[0, 3] = best_ct_origin[0]
        affine[1, 3] = best_ct_origin[1]
        affine[2, 3] = best_ct_origin[2]
    elif viz_needed:
        print("[WARN] Missing CT origin/spacing; skipping 3D visualization.")
        viz_needed = False

    if args.visualize_tandem and viz_needed:
        print("[INFO] Visualizing final straight needles + organs...")
        visualize_bent_needles(
            structure_mask=best_structure_mask,
            label_mapping=best_label_mapping,
            anatomical_library=final_library,
            affine=affine,
            voxel_spacing=best_voxel_spacing,
            tandem_paths=tandem_paths_for_viz,
            ovoid_paths=ovoid_paths_for_viz,
        )

    if args.save_plan_viz and final_library and viz_needed:
        screenshot_path = os.path.join(SAVE_DIR, "plan_3d.png")
        print(f"[INFO] Saving plan visualization to {screenshot_path}")
        visualize_bent_needles(
            structure_mask=best_structure_mask,
            label_mapping=best_label_mapping,
            anatomical_library=final_library,
            affine=affine,
            voxel_spacing=best_voxel_spacing,
            tandem_paths=tandem_paths_for_viz,
            ovoid_paths=ovoid_paths_for_viz,
            screenshot_path=screenshot_path,
            off_screen=True,
        )
