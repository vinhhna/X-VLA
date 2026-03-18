#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.libero_pro.runtime import prepare_libero_runtime

MANIFEST_VERSION = 2
DEFAULT_MINI_SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
DEFAULT_MINI_PERTURBATIONS = ["position", "semantic"]
DEFAULT_MINI_SEEDS = [42, 43]
FULL_SUITES = DEFAULT_MINI_SUITES
FULL_PERTURBATIONS = ["position", "semantic"]
FULL_TASK_COUNT = 10
FULL_BASE_SEED = 42
FULL_EPISODE_COUNT = 50
FULL_INIT_STATE_COUNT = 50
FULL_BENCHMARK_SCOPE = "validated_swap_lan_suites"
DEFAULT_STEPS = 10
DEFAULT_VIDEO_POLICY = "failures_only"
DEFAULT_SERVER_MODE = "managed"

PERTURBATION_RUNTIME_SUFFIX = {
    "position": "swap",
    "object": "object",
    "semantic": "lan",
    "task": "task",
    "environment": "env",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LIBERO-Pro evaluation manifest.")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["smoke", "mini", "full"],
        default="smoke",
        help="Which phase manifest to build.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for manifests and chunk outputs.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default=None,
        help="Optional manifest filename under --output_root. Defaults to <phase>_manifest.json.",
    )
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
        "--base_seed",
        type=int,
        default=FULL_BASE_SEED,
        help="Base seed used when building a full manifest.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Default denoising steps stored in a full manifest.",
    )
    parser.add_argument(
        "--video_policy",
        type=str,
        choices=["none", "all", "failures_only"],
        default=DEFAULT_VIDEO_POLICY,
        help="Default video policy stored in a full manifest.",
    )
    parser.add_argument(
        "--server_mode",
        type=str,
        choices=["managed", "external"],
        default=DEFAULT_SERVER_MODE,
        help="Default server mode stored in a full manifest.",
    )
    return parser.parse_args()


def chunk_id_for(phase: str, suite: str, task_id: int, perturbation_type: str, seed: int) -> str:
    return f"{phase}-{suite}-task{task_id}-{perturbation_type}-seed{seed}"


def load_benchmark_dict(*, libero_pro_root: str | None, libero_config_path: str | None):
    prepare_libero_runtime(
        libero_pro_root=libero_pro_root,
        config_root=libero_config_path,
        strict=False,
    )
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()


def try_resolve_tasks(
    base_suite: str,
    perturbation_type: str,
    *,
    libero_pro_root: str | None,
    libero_config_path: str | None,
) -> List[Any]:
    runtime_suite = f"{base_suite}_{PERTURBATION_RUNTIME_SUFFIX[perturbation_type]}"
    try:
        benchmark_dict = load_benchmark_dict(
            libero_pro_root=libero_pro_root,
            libero_config_path=libero_config_path,
        )
        task_suite = benchmark_dict[runtime_suite]()
        return [task_suite.get_task(i) for i in range(task_suite.get_num_tasks())]
    except Exception:
        return []


def resolve_task_names(
    base_suite: str,
    *,
    libero_pro_root: str | None,
    libero_config_path: str | None,
) -> List[str]:
    for perturbation_type in ("position", "semantic"):
        tasks = try_resolve_tasks(
            base_suite,
            perturbation_type,
            libero_pro_root=libero_pro_root,
            libero_config_path=libero_config_path,
        )
        if tasks:
            return [task.name for task in tasks]
    return []


def build_chunk(
    *,
    phase: str,
    output_root: Path,
    suite: str,
    task_id: int,
    task_name: str,
    perturbation_type: str,
    seed: int,
    episode_count: int,
    init_state_count: int,
) -> Dict[str, Any]:
    chunk_id = chunk_id_for(phase, suite, task_id, perturbation_type, seed)
    chunk_output_dir = output_root / "chunks" / chunk_id
    return {
        "chunk_id": chunk_id,
        "phase": phase,
        "suite": suite,
        "task_id": task_id,
        "task_name": task_name,
        "perturbation_type": perturbation_type,
        "seed": seed,
        "episode_count": episode_count,
        "init_state_count": init_state_count,
        "status": "pending",
        "output_dir": str(chunk_output_dir),
        "error": None,
        "success_rate": None,
        "started_at": None,
        "finished_at": None,
        "attempt_count": 0,
        "last_duration_seconds": None,
    }


def build_smoke_manifest(
    output_root: Path,
    *,
    libero_pro_root: str | None,
    libero_config_path: str | None,
) -> Dict[str, Any]:
    tasks = try_resolve_tasks(
        "libero_spatial",
        "position",
        libero_pro_root=libero_pro_root,
        libero_config_path=libero_config_path,
    )
    task_name = tasks[0].name if tasks else ""
    chunks = [
        build_chunk(
            phase="smoke",
            output_root=output_root,
            suite="libero_spatial",
            task_id=0,
            task_name=task_name,
            perturbation_type="position",
            seed=42,
            episode_count=1,
            init_state_count=1,
        )
    ]
    return {
        "manifest_version": MANIFEST_VERSION,
        "phase": "smoke",
        "created_at": utc_now(),
        "chunks": chunks,
    }


def build_mini_manifest(
    output_root: Path,
    *,
    libero_pro_root: str | None,
    libero_config_path: str | None,
) -> Dict[str, Any]:
    chunks: List[Dict[str, Any]] = []
    for suite in DEFAULT_MINI_SUITES:
        tasks = try_resolve_tasks(
            suite,
            "position",
            libero_pro_root=libero_pro_root,
            libero_config_path=libero_config_path,
        )
        task_count = min(2, len(tasks)) if tasks else 2
        for task_id in range(task_count):
            task_name = tasks[task_id].name if task_id < len(tasks) else ""
            for perturbation_type in DEFAULT_MINI_PERTURBATIONS:
                for seed in DEFAULT_MINI_SEEDS:
                    chunks.append(
                        build_chunk(
                            phase="mini",
                            output_root=output_root,
                            suite=suite,
                            task_id=task_id,
                            task_name=task_name,
                            perturbation_type=perturbation_type,
                            seed=seed,
                            episode_count=2,
                            init_state_count=2,
                        )
                    )

    return {
        "manifest_version": MANIFEST_VERSION,
        "phase": "mini",
        "created_at": utc_now(),
        "chunks": chunks,
    }


def build_manifest(
    phase: str,
    output_root: Path,
    *,
    libero_pro_root: str | None,
    libero_config_path: str | None,
    base_seed: int = FULL_BASE_SEED,
    steps: int = DEFAULT_STEPS,
    video_policy: str = DEFAULT_VIDEO_POLICY,
    server_mode: str = DEFAULT_SERVER_MODE,
) -> Dict[str, Any]:
    if phase == "smoke":
        return build_smoke_manifest(
            output_root,
            libero_pro_root=libero_pro_root,
            libero_config_path=libero_config_path,
        )
    if phase == "mini":
        return build_mini_manifest(
            output_root,
            libero_pro_root=libero_pro_root,
            libero_config_path=libero_config_path,
        )
    if phase == "full":
        chunks: List[Dict[str, Any]] = []
        for suite in FULL_SUITES:
            task_names = resolve_task_names(
                suite,
                libero_pro_root=libero_pro_root,
                libero_config_path=libero_config_path,
            )
            for task_id in range(FULL_TASK_COUNT):
                task_name = task_names[task_id] if task_id < len(task_names) else ""
                for perturbation_type in FULL_PERTURBATIONS:
                    chunks.append(
                        build_chunk(
                            phase="full",
                            output_root=output_root,
                            suite=suite,
                            task_id=task_id,
                            task_name=task_name,
                            perturbation_type=perturbation_type,
                            seed=base_seed,
                            episode_count=FULL_EPISODE_COUNT,
                            init_state_count=FULL_INIT_STATE_COUNT,
                        )
                    )

        return {
            "manifest_version": MANIFEST_VERSION,
            "phase": "full",
            "created_at": utc_now(),
            "benchmark_scope": FULL_BENCHMARK_SCOPE,
            "base_seed": int(base_seed),
            "steps": int(steps),
            "video_policy": video_policy,
            "server_mode": server_mode,
            "eta_priors": {},
            "chunks": chunks,
        }
    raise ValueError(f"Unsupported phase: {phase}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        args.phase,
        output_root,
        libero_pro_root=args.libero_pro_root,
        libero_config_path=args.libero_config_path,
        base_seed=args.base_seed,
        steps=args.steps,
        video_policy=args.video_policy,
        server_mode=args.server_mode,
    )
    manifest_name = args.manifest_name or f"{args.phase}_manifest.json"
    manifest_path = output_root / manifest_name
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"wrote manifest to: {manifest_path}")
    print(f"chunk count: {len(manifest['chunks'])}")
    if manifest["chunks"]:
        print(f"first chunk id: {manifest['chunks'][0]['chunk_id']}")


if __name__ == "__main__":
    main()
