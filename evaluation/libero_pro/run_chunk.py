#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.libero_pro.runtime import prepare_libero_runtime

PERTURBATION_RUNTIME_SUFFIX = {
    "position": "swap",
    "object": "object",
    "semantic": "lan",
    "task": "task",
    "environment": "env",
}

BASE_SUITE_HORIZON = {
    "libero_goal": 300,
    "libero_spatial": 220,
    "libero_10": 520,
    "libero_object": 280,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LIBERO-Pro chunk.")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to the manifest JSON.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_next", action="store_true", help="Run the first pending chunk in the manifest.")
    group.add_argument("--chunk_id", type=str, help="Run a specific chunk by id.")

    parser.add_argument(
        "--connection_info",
        type=str,
        default=None,
        help="Path to deploy.py info.json containing host and port.",
    )
    parser.add_argument("--server_ip", type=str, default=None, help="Manual server IP.")
    parser.add_argument("--server_port", type=int, default=None, help="Manual server port.")
    parser.add_argument("--steps", type=int, default=10, help="Denoising steps passed to the policy server.")
    parser.add_argument(
        "--libero_pro_root",
        type=str,
        default=None,
        help="Optional path to a local LIBERO-Pro checkout for runtime discovery.",
    )
    parser.add_argument(
        "--libero_config_path",
        type=str,
        default=None,
        help="Optional config root to use instead of ~/.libero_xvla.",
    )
    parser.add_argument(
        "--video_policy",
        type=str,
        choices=["none", "all", "failures_only"],
        default=None,
        help="Video retention policy. Defaults to --save_videos/--no_save_videos behavior.",
    )
    parser.add_argument(
        "--save_videos",
        dest="save_videos",
        action="store_true",
        help="Backward-compatible alias for --video_policy all.",
    )
    parser.add_argument(
        "--no_save_videos",
        dest="save_videos",
        action="store_false",
        help="Backward-compatible alias for --video_policy none.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Abort immediately on rollout errors. Useful for smoke testing.",
    )
    parser.set_defaults(save_videos=True)
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def resolve_server(args: argparse.Namespace) -> tuple[str, int]:
    if args.connection_info:
        with open(args.connection_info, "r", encoding="utf-8") as f:
            info = json.load(f)
        return info["host"], int(info["port"])

    if args.server_ip is None or args.server_port is None:
        raise ValueError("Specify either --connection_info or both --server_ip and --server_port.")
    return args.server_ip, int(args.server_port)


def select_chunk(manifest: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    chunks = manifest.get("chunks", [])
    if args.run_next:
        for chunk in chunks:
            if chunk.get("status") == "pending":
                return chunk
        raise RuntimeError("No pending chunks found in the manifest.")

    for chunk in chunks:
        if chunk.get("chunk_id") == args.chunk_id:
            status = chunk.get("status")
            if status == "running":
                raise RuntimeError(f"Chunk {args.chunk_id} is already marked as running.")
            return chunk
    raise RuntimeError(f"Chunk {args.chunk_id} not found in the manifest.")


def ensure_runtime_suite(base_suite: str, perturbation_type: str) -> str:
    suffix = PERTURBATION_RUNTIME_SUFFIX[perturbation_type]
    return f"{base_suite}_{suffix}"


def load_libero_runtime() -> tuple[Any, Any]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, get_libero_path, OffScreenRenderEnv


def build_smoke_client(host: str, port: int, steps: int):
    from evaluation.libero.libero_client import ClientModel

    class SmokeClientModel(ClientModel):
        def __init__(self, host: str, port: int, steps: int):
            super().__init__(host, port)
            self.steps = int(steps)

        def _format_query(self, obs: Dict, goal: str) -> Dict:
            payload = super()._format_query(obs, goal)
            payload["steps"] = self.steps
            return payload

    return SmokeClientModel(host, port, steps)


def load_libero_helpers():
    from evaluation.libero.libero_client import LiberoAbsActionProcessor, _flip_agentview

    return LiberoAbsActionProcessor, _flip_agentview


def init_env(task_suite, task_id: int, seed: int, init_state_index: int, act_type: str = "abs") -> tuple[Any, str, Dict[str, Any]]:
    benchmark, get_libero_path, OffScreenRenderEnv = load_libero_runtime()
    del benchmark  # imported for consistency; task_suite is already constructed.

    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed + 100)
    obs = env.reset()

    init_states = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[init_state_index])

    for _ in range(10):
        settle_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, _, _, _ = env.step(settle_action)

    if act_type == "abs":
        for robot in env.env.robots:
            robot.controller.use_delta = False
    elif act_type != "rel":
        raise ValueError("act_type must be 'abs' or 'rel'")

    return env, task.language, obs


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def resolve_video_policy(args: argparse.Namespace) -> str:
    if args.video_policy is not None:
        return args.video_policy
    return "all" if args.save_videos else "none"


def cleanup_chunk_artifacts(chunk_dir: Path) -> None:
    for artifact_name in ("error.txt", "rollout.json", "summary.json"):
        artifact_path = chunk_dir / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()

    for video_path in chunk_dir.glob("rollout_ep*.mp4"):
        video_path.unlink()


def save_rollout_video(chunk_dir: Path, episode_index: int, images: list[np.ndarray], should_save: bool) -> Optional[Path]:
    if not should_save or not images:
        return None

    video_path = chunk_dir / f"rollout_ep{episode_index}.mp4"
    imageio.mimsave(video_path.as_posix(), images, fps=30)
    return video_path


def seconds_between(started_at: Optional[str], finished_at: Optional[str]) -> Optional[float]:
    if not started_at or not finished_at:
        return None
    start_dt = datetime.fromisoformat(started_at)
    finish_dt = datetime.fromisoformat(finished_at)
    return round((finish_dt - start_dt).total_seconds(), 6)


def rollout_chunk(
    *,
    chunk: Dict[str, Any],
    host: str,
    port: int,
    steps: int,
    video_policy: str,
    fail_fast: bool,
) -> Dict[str, Any]:
    benchmark, _, _ = load_libero_runtime()
    benchmark_dict = benchmark.get_benchmark_dict()
    processor_cls, flip_agentview = load_libero_helpers()
    processor = processor_cls()
    runtime_suite = ensure_runtime_suite(chunk["suite"], chunk["perturbation_type"])
    task_suite = benchmark_dict[runtime_suite]()
    task = task_suite.get_task(chunk["task_id"])
    chunk["task_name"] = task.name

    horizon = BASE_SUITE_HORIZON[chunk["suite"]]
    init_states = task_suite.get_task_init_states(chunk["task_id"])
    init_state_count = min(int(chunk["init_state_count"]), int(init_states.shape[0]))
    episode_count = min(int(chunk["episode_count"]), init_state_count)

    client = build_smoke_client(host, port, steps=steps)

    chunk_dir = Path(chunk["output_dir"])
    chunk_dir.mkdir(parents=True, exist_ok=True)

    rollouts = []
    success_total = 0.0
    fatal_error: Optional[str] = None
    error_count = 0
    capture_frames = video_policy != "none"

    for ep in range(episode_count):
        client.reset()
        images = []
        done_flag = False
        env = None
        task_name = task.name
        try:
            env, language, obs = init_env(
                task_suite=task_suite,
                task_id=int(chunk["task_id"]),
                seed=int(chunk["seed"]) + ep,
                init_state_index=ep,
                act_type="abs",
            )

            for step_idx in range(horizon):
                robo_ori = processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
                robo_pos = env.env.robots[0].controller.ee_pos
                obs["robo_ori"] = robo_ori
                obs["robo_pos"] = robo_pos

                if capture_frames:
                    images.append(flip_agentview(obs["agentview_image"]))

                executed_action = client.step(obs, language)
                obs, _, done, _ = env.step(executed_action)
                if done:
                    done_flag = True
                    break

            success = 1.0 if done_flag else 0.0
            success_total += success
            should_save_video = video_policy == "all" or (video_policy == "failures_only" and success < 1.0)
            video_path = save_rollout_video(chunk_dir, ep, images, should_save_video)
            rollout_record = {
                "chunk_id": chunk["chunk_id"],
                "suite": chunk["suite"],
                "runtime_suite": runtime_suite,
                "task_id": int(chunk["task_id"]),
                "task_name": task_name,
                "language_instruction": task.language,
                "perturbation_type": chunk["perturbation_type"],
                "seed": int(chunk["seed"]) + ep,
                "episode_index": ep,
                "init_state_index": ep,
                "horizon": horizon,
                "success": success,
                "video_path": str(video_path) if video_path is not None else None,
            }
            rollouts.append(rollout_record)
        except BaseException as e:
            error_count += 1
            should_save_video = video_policy in ("all", "failures_only")
            video_path = save_rollout_video(chunk_dir, ep, images, should_save_video)
            error_record = {
                "chunk_id": chunk["chunk_id"],
                "suite": chunk["suite"],
                "runtime_suite": runtime_suite,
                "task_id": int(chunk["task_id"]),
                "task_name": task_name,
                "perturbation_type": chunk["perturbation_type"],
                "seed": int(chunk["seed"]) + ep,
                "episode_index": ep,
                "init_state_index": ep,
                "horizon": horizon,
                "success": 0.0,
                "error": str(e),
                "video_path": str(video_path) if video_path is not None else None,
            }
            rollouts.append(error_record)
            if fail_fast or isinstance(e, KeyboardInterrupt):
                fatal_error = traceback.format_exc()
                break
        finally:
            if env is not None:
                env.close()

    success_rate = success_total / max(len(rollouts), 1)
    summary = {
        "chunk_id": chunk["chunk_id"],
        "phase": chunk["phase"],
        "suite": chunk["suite"],
        "runtime_suite": runtime_suite,
        "task_id": int(chunk["task_id"]),
        "task_name": task.name,
        "perturbation_type": chunk["perturbation_type"],
        "seed": int(chunk["seed"]),
        "episode_count": episode_count,
        "init_state_count": init_state_count,
        "success_rate": success_rate,
        "completed_rollouts": len(rollouts),
        "error_count": error_count,
    }
    if fatal_error is not None:
        summary["fatal_error"] = fatal_error
    write_json(chunk_dir / "rollout.json", {"rollouts": rollouts})
    write_json(chunk_dir / "summary.json", summary)
    return {
        "summary": summary,
        "fatal_error": fatal_error,
    }


def main() -> None:
    args = parse_args()
    video_policy = resolve_video_policy(args)
    manifest_path = Path(args.manifest_path).resolve()
    prepare_libero_runtime(
        libero_pro_root=args.libero_pro_root,
        config_root=args.libero_config_path,
        strict=True,
    )
    manifest = load_manifest(manifest_path)
    chunk = select_chunk(manifest, args)
    chunk_dir = Path(chunk["output_dir"])
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cleanup_chunk_artifacts(chunk_dir)
    error_path = chunk_dir / "error.txt"

    host, port = resolve_server(args)

    chunk["status"] = "running"
    chunk["error"] = None
    chunk["success_rate"] = None
    chunk["started_at"] = utc_now()
    chunk["finished_at"] = None
    chunk["last_duration_seconds"] = None
    chunk["attempt_count"] = int(chunk.get("attempt_count", 0)) + 1
    save_manifest(manifest_path, manifest)

    try:
        run_result = rollout_chunk(
            chunk=chunk,
            host=host,
            port=port,
            steps=args.steps,
            video_policy=video_policy,
            fail_fast=args.fail_fast,
        )
        summary = run_result["summary"]
        chunk["task_name"] = summary["task_name"]
        chunk["success_rate"] = summary["success_rate"]
        chunk["finished_at"] = utc_now()
        chunk["last_duration_seconds"] = seconds_between(chunk.get("started_at"), chunk.get("finished_at"))

        fatal_error = run_result["fatal_error"]
        if fatal_error is None:
            chunk["status"] = "success"
            save_manifest(manifest_path, manifest)
            print(json.dumps(summary, indent=2))
            return

        chunk["status"] = "failed"
        chunk["error"] = fatal_error
        save_manifest(manifest_path, manifest)
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(chunk["error"])
        print(chunk["error"], file=sys.stderr)
        sys.exit(2)
    except BaseException as exc:
        chunk["status"] = "failed"
        chunk["finished_at"] = utc_now()
        chunk["error"] = traceback.format_exc()
        chunk["last_duration_seconds"] = seconds_between(chunk.get("started_at"), chunk.get("finished_at"))
        save_manifest(manifest_path, manifest)
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(chunk["error"])
        print(chunk["error"], file=sys.stderr)
        if isinstance(exc, KeyboardInterrupt):
            sys.exit(130)
        sys.exit(2)


if __name__ == "__main__":
    main()
