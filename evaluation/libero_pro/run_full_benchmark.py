#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.error import URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.libero_pro.build_manifest import (
    DEFAULT_STEPS,
    DEFAULT_VIDEO_POLICY,
    FULL_BASE_SEED,
    FULL_BENCHMARK_SCOPE,
    FULL_PERTURBATIONS,
    FULL_SUITES,
    build_manifest,
)
from evaluation.libero_pro.run_chunk import load_manifest, save_manifest, utc_now
from evaluation.libero_pro.runtime import prepare_libero_runtime

EXPECTED_RUNTIME_HOURS = "16-18"
BOOKING_WINDOW_HOURS = 24
WORST_CASE_NOTE = "2-3 days if the spatial-semantic slow path or repeated retries reappear."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full LIBERO-Pro benchmark with a managed GPU server.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the X-VLA LIBERO checkpoint.")
    parser.add_argument("--processor_path", type=str, default=None, help="Optional XVLA processor path.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA weights path.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="logs/libero_pro_full",
        help="Directory that stores the full benchmark manifest and artifacts.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="full_manifest.json",
        help="Manifest filename under --output_root.",
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
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Denoising steps passed to the policy server.")
    parser.add_argument(
        "--server_host",
        type=str,
        default="127.0.0.1",
        help="Host used for the managed X-VLA server.",
    )
    parser.add_argument("--server_port", type=int, default=8011, help="Port used for the managed X-VLA server.")
    parser.add_argument("--device", type=str, default="cuda", help="Device passed to deploy.py.")
    parser.add_argument(
        "--video_policy",
        type=str,
        choices=["none", "all", "failures_only"],
        default=DEFAULT_VIDEO_POLICY,
        help="Video policy passed to run_chunk.py.",
    )
    parser.add_argument(
        "--startup_timeout",
        type=int,
        default=900,
        help="Seconds to wait for the managed server to become ready.",
    )
    parser.add_argument(
        "--cleanup_legacy_logs",
        action="store_true",
        help="Delete logs/libero_pro_smoke and logs/libero_pro_mini after ETA priors are captured.",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Reset failed chunks to pending before resuming.",
    )
    parser.add_argument(
        "--rebuild_manifest",
        action="store_true",
        help="Overwrite any existing full manifest and start a new queue.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Optional limit on how many chunks to run in this invocation. Use 0 for prepare-only.",
    )
    return parser.parse_args()


def ensure_libero_pro_env() -> None:
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    prefix_name = Path(sys.prefix).name
    if env_name == "libero_pro" or prefix_name == "libero_pro":
        return
    raise RuntimeError(
        "run_full_benchmark.py must run inside the 'libero_pro' conda environment. "
        f"Detected CONDA_DEFAULT_ENV={env_name!r}, sys.prefix={sys.prefix!r}."
    )


def require_runtime(args: argparse.Namespace) -> None:
    prepare_libero_runtime(
        libero_pro_root=args.libero_pro_root,
        config_root=args.libero_config_path,
        strict=True,
    )
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    required_suites = {f"{suite}_{suffix}" for suite in FULL_SUITES for suffix in ("swap", "lan")}
    missing = sorted(required_suites.difference(benchmark_dict.keys()))
    if missing:
        raise RuntimeError(f"LIBERO-Pro runtime is missing benchmark suites: {missing}")


def nested_get(mapping: Dict[str, Any], *keys: str) -> Optional[float]:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    return float(current)


def build_eta_priors(mini_manifest_path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not mini_manifest_path.exists():
        return {}

    manifest = load_manifest(mini_manifest_path)
    grouped: Dict[tuple[str, str], list[float]] = defaultdict(list)

    for chunk in manifest.get("chunks", []):
        if chunk.get("status") != "success":
            continue
        started_at = chunk.get("started_at")
        finished_at = chunk.get("finished_at")
        episode_count = int(chunk.get("episode_count") or 0)
        if not started_at or not finished_at or episode_count <= 0:
            continue
        start_sec = datetime.fromisoformat(started_at).timestamp()
        finish_sec = datetime.fromisoformat(finished_at).timestamp()
        grouped[(chunk["suite"], chunk["perturbation_type"])].append((finish_sec - start_sec) / episode_count)

    priors: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for suite in FULL_SUITES:
        suite_priors: Dict[str, Dict[str, Any]] = {}
        for perturbation_type in FULL_PERTURBATIONS:
            values = sorted(grouped.get((suite, perturbation_type), []))
            if not values:
                continue

            if suite == "libero_spatial" and perturbation_type == "semantic" and len(values) > 1:
                expected_seconds = min(values)
                excluded_outlier = max(values)
            elif len(values) > 2:
                expected_seconds = statistics.mean(values[1:-1])
                excluded_outlier = None
            else:
                expected_seconds = statistics.mean(values)
                excluded_outlier = None

            suite_priors[perturbation_type] = {
                "seconds_per_episode": round(expected_seconds, 6),
                "conservative_seconds_per_episode": round(max(values), 6),
                "sample_count": len(values),
                "source": "mini_manifest",
                "excluded_outlier_seconds_per_episode": (
                    round(excluded_outlier, 6) if excluded_outlier is not None else None
                ),
            }
        if suite_priors:
            priors[suite] = suite_priors
    return priors


def cleanup_legacy_logs(root: Path) -> None:
    for relative_path in ("logs/libero_pro_smoke", "logs/libero_pro_mini"):
        target = root / relative_path
        if target.exists():
            shutil.rmtree(target)


def prepare_manifest(args: argparse.Namespace, output_root: Path, manifest_path: Path) -> Dict[str, Any]:
    mini_manifest_path = REPO_ROOT / "logs" / "libero_pro_mini" / "mini_manifest.json"
    eta_priors = build_eta_priors(mini_manifest_path)

    if args.cleanup_legacy_logs:
        cleanup_legacy_logs(REPO_ROOT)

    if manifest_path.exists() and not args.rebuild_manifest:
        manifest = load_manifest(manifest_path)
        if manifest.get("phase") != "full":
            raise RuntimeError(f"Manifest at {manifest_path} is not a full LIBERO-Pro manifest.")
    else:
        manifest = build_manifest(
            "full",
            output_root,
            libero_pro_root=args.libero_pro_root,
            libero_config_path=args.libero_config_path,
            base_seed=FULL_BASE_SEED,
            steps=args.steps,
            video_policy=args.video_policy,
            server_mode="managed",
        )

    manifest["benchmark_scope"] = FULL_BENCHMARK_SCOPE
    manifest["base_seed"] = FULL_BASE_SEED
    manifest["steps"] = int(args.steps)
    manifest["video_policy"] = args.video_policy
    manifest["server_mode"] = "managed"
    if not manifest.get("eta_priors") and eta_priors:
        manifest["eta_priors"] = eta_priors

    stale_count = 0
    retry_count = 0
    for chunk in manifest.get("chunks", []):
        if chunk.get("status") == "running":
            chunk["status"] = "pending"
            chunk["started_at"] = None
            chunk["finished_at"] = None
            chunk["error"] = None
            chunk["success_rate"] = None
            chunk["last_duration_seconds"] = None
            stale_count += 1
        if args.retry_failed and chunk.get("status") == "failed":
            chunk["status"] = "pending"
            chunk["started_at"] = None
            chunk["finished_at"] = None
            chunk["error"] = None
            chunk["success_rate"] = None
            chunk["last_duration_seconds"] = None
            retry_count += 1

    save_manifest(manifest_path, manifest)
    if stale_count:
        print(f"reset {stale_count} stale running chunk(s) to pending")
    if retry_count:
        print(f"reset {retry_count} failed chunk(s) to pending")
    return manifest


def is_port_busy(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def probe_server(host: str, port: int) -> bool:
    try:
        with urlopen(f"http://{host}:{port}/openapi.json", timeout=3) as response:
            return response.status == 200
    except (URLError, TimeoutError, OSError):
        return False


def start_server(args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    if is_port_busy(args.server_host, args.server_port):
        raise RuntimeError(
            f"Port {args.server_port} on {args.server_host} is already in use. "
            "Stop the conflicting process or choose a different --server_port."
        )

    server_dir = output_root / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    info_path = server_dir / "info.json"
    if info_path.exists():
        info_path.unlink()

    stdout_path = server_dir / "server.stdout.log"
    stderr_path = server_dir / "server.stderr.log"
    stdout_handle = open(stdout_path, "a", encoding="utf-8")
    stderr_handle = open(stderr_path, "a", encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "deploy",
        "--model_path",
        args.model_path,
        "--output_dir",
        str(server_dir),
        "--port",
        str(args.server_port),
        "--host",
        args.server_host,
        "--device",
        args.device,
        "--disable_slurm",
    ]
    if args.processor_path:
        command.extend(["--processor_path", args.processor_path])
    if args.lora_path:
        command.extend(["--LoRA_path", args.lora_path])

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    return {
        "process": process,
        "info_path": info_path,
        "stdout_handle": stdout_handle,
        "stderr_handle": stderr_handle,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "server_dir": server_dir,
        "host": args.server_host,
        "port": args.server_port,
    }


def wait_for_server_ready(server_state: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    process: subprocess.Popen[str] = server_state["process"]
    info_path: Path = server_state["info_path"]
    host = server_state["host"]
    port = server_state["port"]
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Managed X-VLA server exited before becoming ready. "
                f"See {server_state['stdout_path']} and {server_state['stderr_path']}."
            )

        if info_path.exists() and probe_server(host, port):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            server_state["server_info"] = info
            return info

        time.sleep(1.0)

    raise TimeoutError(
        f"Timed out after {timeout_seconds}s waiting for the managed server at {host}:{port}. "
        f"See {server_state['stdout_path']} and {server_state['stderr_path']}."
    )


def stop_server(server_state: Optional[Dict[str, Any]]) -> None:
    if not server_state:
        return

    process: subprocess.Popen[str] = server_state["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    server_state["stdout_handle"].close()
    server_state["stderr_handle"].close()


def chunk_rollout_count(chunk: Dict[str, Any]) -> int:
    summary_path = Path(chunk["output_dir"]) / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return int(summary.get("completed_rollouts") or 0)
    return 0


def chunk_group_key(chunk: Dict[str, Any]) -> tuple[str, str]:
    return chunk["suite"], chunk["perturbation_type"]


def build_group_estimates(manifest: Dict[str, Any]) -> Dict[tuple[str, str], float]:
    priors = manifest.get("eta_priors", {})
    observed: Dict[tuple[str, str], list[float]] = defaultdict(list)

    for chunk in manifest.get("chunks", []):
        if chunk.get("status") != "success":
            continue
        duration = chunk.get("last_duration_seconds")
        episode_count = int(chunk.get("episode_count") or 0)
        if duration is None or episode_count <= 0:
            continue
        observed[chunk_group_key(chunk)].append(float(duration) / episode_count)

    estimates: Dict[tuple[str, str], float] = {}
    for suite in FULL_SUITES:
        for perturbation_type in FULL_PERTURBATIONS:
            key = (suite, perturbation_type)
            if observed.get(key):
                estimates[key] = statistics.mean(observed[key])
                continue

            prior_seconds = nested_get(priors, suite, perturbation_type, "seconds_per_episode")
            if prior_seconds is not None:
                estimates[key] = prior_seconds

    return estimates


def chunk_success_rollouts(chunk: Dict[str, Any]) -> float:
    success_rate = chunk.get("success_rate")
    if success_rate is None:
        return 0.0
    completed_rollouts = chunk_rollout_count(chunk)
    return float(success_rate) * completed_rollouts


def generate_results(manifest: Dict[str, Any]) -> Dict[str, Any]:
    chunks = manifest.get("chunks", [])
    terminal_chunks = [chunk for chunk in chunks if chunk.get("status") in {"success", "failed"}]
    success_chunks = [chunk for chunk in chunks if chunk.get("status") == "success"]
    failed_chunks = [chunk for chunk in chunks if chunk.get("status") == "failed"]
    running_chunks = [chunk for chunk in chunks if chunk.get("status") == "running"]
    pending_chunks = [chunk for chunk in chunks if chunk.get("status") == "pending"]

    completed_rollouts = sum(chunk_rollout_count(chunk) for chunk in chunks)
    total_rollouts = sum(int(chunk.get("episode_count") or 0) for chunk in chunks)
    average_chunk_duration = None
    completed_durations = [float(chunk["last_duration_seconds"]) for chunk in success_chunks if chunk.get("last_duration_seconds") is not None]
    if completed_durations:
        average_chunk_duration = round(statistics.mean(completed_durations), 6)

    group_estimates = build_group_estimates(manifest)
    remaining_seconds = 0.0
    for chunk in chunks:
        if chunk.get("status") not in {"pending", "running"}:
            continue
        per_episode_seconds = group_estimates.get(chunk_group_key(chunk))
        if per_episode_seconds is None:
            continue
        remaining_seconds += per_episode_seconds * int(chunk.get("episode_count") or 0)
    remaining_seconds *= 1.1

    created_at = manifest.get("created_at")
    elapsed_seconds = None
    if created_at:
        created_epoch = datetime.fromisoformat(created_at).timestamp()
        elapsed_seconds = max(time.time() - created_epoch, 0.0)

    eta_timestamp = None
    if remaining_seconds:
        eta_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() + remaining_seconds))

    suite_rows = []
    suite_aggregates: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "chunk_count": 0,
            "completed_chunks": 0,
            "success_chunks": 0,
            "failed_chunks": 0,
            "completed_rollouts": 0,
            "success_rollouts": 0.0,
            "duration_seconds": [],
        }
    )
    for chunk in chunks:
        aggregate = suite_aggregates[chunk_group_key(chunk)]
        aggregate["chunk_count"] += 1
        aggregate["completed_rollouts"] += chunk_rollout_count(chunk)
        if chunk.get("status") in {"success", "failed"}:
            aggregate["completed_chunks"] += 1
        if chunk.get("status") == "success":
            aggregate["success_chunks"] += 1
            aggregate["success_rollouts"] += chunk_success_rollouts(chunk)
            if chunk.get("last_duration_seconds") is not None:
                aggregate["duration_seconds"].append(float(chunk["last_duration_seconds"]))
        if chunk.get("status") == "failed":
            aggregate["failed_chunks"] += 1

    for (suite, perturbation_type), aggregate in sorted(suite_aggregates.items()):
        completed_rollouts_count = aggregate["completed_rollouts"]
        success_rate = None
        if completed_rollouts_count:
            success_rate = round(aggregate["success_rollouts"] / completed_rollouts_count, 6)
        suite_rows.append(
            {
                "suite": suite,
                "perturbation_type": perturbation_type,
                "chunk_count": aggregate["chunk_count"],
                "completed_chunks": aggregate["completed_chunks"],
                "success_chunks": aggregate["success_chunks"],
                "failed_chunks": aggregate["failed_chunks"],
                "completed_rollouts": completed_rollouts_count,
                "success_rate": success_rate,
                "mean_duration_seconds": (
                    round(statistics.mean(aggregate["duration_seconds"]), 6)
                    if aggregate["duration_seconds"]
                    else None
                ),
            }
        )

    task_rows = []
    for chunk in sorted(chunks, key=lambda item: (item["suite"], item["task_id"], item["perturbation_type"])):
        task_rows.append(
            {
                "suite": chunk["suite"],
                "perturbation_type": chunk["perturbation_type"],
                "task_id": chunk["task_id"],
                "task_name": chunk.get("task_name") or "",
                "chunk_id": chunk["chunk_id"],
                "status": chunk.get("status"),
                "attempt_count": chunk.get("attempt_count", 0),
                "completed_rollouts": chunk_rollout_count(chunk),
                "success_rate": chunk.get("success_rate"),
                "last_duration_seconds": chunk.get("last_duration_seconds"),
                "started_at": chunk.get("started_at"),
                "finished_at": chunk.get("finished_at"),
            }
        )

    chunk_rows = []
    for chunk in chunks:
        chunk_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "status": chunk.get("status"),
                "suite": chunk["suite"],
                "perturbation_type": chunk["perturbation_type"],
                "task_id": chunk["task_id"],
                "task_name": chunk.get("task_name") or "",
                "seed": chunk.get("seed"),
                "episode_count": chunk.get("episode_count"),
                "init_state_count": chunk.get("init_state_count"),
                "attempt_count": chunk.get("attempt_count", 0),
                "completed_rollouts": chunk_rollout_count(chunk),
                "success_rate": chunk.get("success_rate"),
                "last_duration_seconds": chunk.get("last_duration_seconds"),
                "started_at": chunk.get("started_at"),
                "finished_at": chunk.get("finished_at"),
                "output_dir": chunk.get("output_dir"),
            }
        )

    return {
        "updated_at": utc_now(),
        "counts": {
            "total_chunks": len(chunks),
            "completed_chunks": len(terminal_chunks),
            "success_chunks": len(success_chunks),
            "failed_chunks": len(failed_chunks),
            "pending_chunks": len(pending_chunks),
            "running_chunks": len(running_chunks),
            "total_rollouts": total_rollouts,
            "completed_rollouts": completed_rollouts,
        },
        "timing": {
            "elapsed_seconds": round(elapsed_seconds, 6) if elapsed_seconds is not None else None,
            "average_chunk_duration_seconds": average_chunk_duration,
            "estimated_remaining_seconds": round(remaining_seconds, 6) if remaining_seconds else 0.0,
            "estimated_total_seconds": (
                round((elapsed_seconds or 0.0) + remaining_seconds, 6) if elapsed_seconds is not None else None
            ),
            "eta_timestamp": eta_timestamp,
            "expected_runtime_hours": EXPECTED_RUNTIME_HOURS,
            "booking_window_hours": BOOKING_WINDOW_HOURS,
            "worst_case_note": WORST_CASE_NOTE,
        },
        "chunks": chunk_rows,
        "suites": suite_rows,
        "tasks": task_rows,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    path: Path,
    manifest_path: Path,
    output_root: Path,
    progress: Dict[str, Any],
    current_chunk_id: Optional[str],
    server_state: Optional[Dict[str, Any]],
) -> None:
    counts = progress["counts"]
    timing = progress["timing"]
    lines = [
        "# LIBERO-Pro Full Benchmark",
        "",
        f"- Updated at: {progress['updated_at']}",
        f"- Manifest: {manifest_path}",
        f"- Output root: {output_root}",
        f"- Current chunk: {current_chunk_id or 'idle'}",
        f"- Total chunks: {counts['total_chunks']}",
        f"- Completed chunks: {counts['completed_chunks']}",
        f"- Success chunks: {counts['success_chunks']}",
        f"- Failed chunks: {counts['failed_chunks']}",
        f"- Pending chunks: {counts['pending_chunks']}",
        f"- Running chunks: {counts['running_chunks']}",
        f"- Completed rollouts: {counts['completed_rollouts']}/{counts['total_rollouts']}",
        f"- Average chunk duration (s): {timing['average_chunk_duration_seconds']}",
        f"- Estimated remaining (s): {timing['estimated_remaining_seconds']}",
        f"- ETA: {timing['eta_timestamp']}",
        f"- Expected runtime (hours): {timing['expected_runtime_hours']}",
        f"- Booking window (hours): {timing['booking_window_hours']}",
        f"- Worst-case note: {timing['worst_case_note']}",
    ]

    if server_state:
        lines.extend(
            [
                "",
                "## Server",
                "",
                f"- Host: {server_state['host']}",
                f"- Port: {server_state['port']}",
                f"- Info path: {server_state['info_path']}",
                f"- Stdout log: {server_state['stdout_path']}",
                f"- Stderr log: {server_state['stderr_path']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Suite Summary",
            "",
            "| Suite | Perturbation | Completed | Success | Failed | Rollouts | Success Rate | Mean Duration (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in progress["suites"]:
        lines.append(
            f"| {row['suite']} | {row['perturbation_type']} | {row['completed_chunks']}/{row['chunk_count']} | "
            f"{row['success_chunks']} | {row['failed_chunks']} | {row['completed_rollouts']} | "
            f"{row['success_rate']} | {row['mean_duration_seconds']} |"
        )

    failed_rows = [row for row in progress["chunks"] if row["status"] == "failed"]
    if failed_rows:
        lines.extend(["", "## Failed Chunks", ""])
        for row in failed_rows:
            lines.append(f"- {row['chunk_id']} ({row['suite']} / {row['perturbation_type']} / task {row['task_id']})")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def write_reports(
    output_root: Path,
    manifest_path: Path,
    manifest: Dict[str, Any],
    current_chunk_id: Optional[str],
    server_state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    progress = generate_results(manifest)
    progress["phase"] = "full"
    progress["manifest_path"] = str(manifest_path)
    progress["output_root"] = str(output_root)
    progress["current_chunk"] = current_chunk_id
    progress["server"] = (
        {
            "host": server_state["host"],
            "port": server_state["port"],
            "info_path": str(server_state["info_path"]),
            "stdout_path": str(server_state["stdout_path"]),
            "stderr_path": str(server_state["stderr_path"]),
        }
        if server_state
        else None
    )

    with open(output_root / "progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
        f.write("\n")

    write_csv(output_root / "results_by_chunk.csv", progress["chunks"])
    write_csv(output_root / "results_by_suite.csv", progress["suites"])
    write_csv(output_root / "results_by_task.csv", progress["tasks"])
    write_summary_markdown(output_root / "summary.md", manifest_path, output_root, progress, current_chunk_id, server_state)
    return progress


def next_pending_chunk(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for chunk in manifest.get("chunks", []):
        if chunk.get("status") == "pending":
            return chunk
    return None


def ensure_chunk_failure_recorded(manifest_path: Path, chunk_id: str, return_code: int) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    for chunk in manifest.get("chunks", []):
        if chunk.get("chunk_id") != chunk_id:
            continue
        if chunk.get("status") == "running":
            chunk["status"] = "failed"
            chunk["finished_at"] = utc_now()
            if chunk.get("started_at") and chunk.get("finished_at"):
                start_sec = datetime.fromisoformat(chunk["started_at"]).timestamp()
                finish_sec = datetime.fromisoformat(chunk["finished_at"]).timestamp()
                chunk["last_duration_seconds"] = round(max(finish_sec - start_sec, 0.0), 6)
            chunk["error"] = (
                f"run_chunk.py exited with return code {return_code} before it updated the manifest."
            )
            save_manifest(manifest_path, manifest)
        return manifest
    return manifest


def run_single_chunk(args: argparse.Namespace, manifest_path: Path, chunk_id: str) -> int:
    command = [
        sys.executable,
        str(REPO_ROOT / "evaluation" / "libero_pro" / "run_chunk.py"),
        "--manifest_path",
        str(manifest_path),
        "--chunk_id",
        chunk_id,
        "--server_ip",
        args.server_host,
        "--server_port",
        str(args.server_port),
        "--steps",
        str(args.steps),
        "--video_policy",
        args.video_policy,
    ]
    if args.libero_pro_root:
        command.extend(["--libero_pro_root", args.libero_pro_root])
    if args.libero_config_path:
        command.extend(["--libero_config_path", args.libero_config_path])

    result = subprocess.run(command, cwd=REPO_ROOT)
    return result.returncode


def print_progress(progress: Dict[str, Any]) -> None:
    counts = progress["counts"]
    timing = progress["timing"]
    print(
        "[progress] "
        f"chunks={counts['completed_chunks']}/{counts['total_chunks']} "
        f"success={counts['success_chunks']} failed={counts['failed_chunks']} "
        f"rollouts={counts['completed_rollouts']}/{counts['total_rollouts']} "
        f"avg_chunk_s={timing['average_chunk_duration_seconds']} "
        f"remaining_s={timing['estimated_remaining_seconds']} "
        f"eta={timing['eta_timestamp']}"
    )


def main() -> None:
    args = parse_args()
    ensure_libero_pro_env()
    require_runtime(args)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name

    manifest = prepare_manifest(args, output_root, manifest_path)
    progress = write_reports(output_root, manifest_path, manifest, current_chunk_id=None, server_state=None)
    print_progress(progress)

    if args.max_chunks == 0:
        print("max_chunks=0, preparation complete without executing any chunks")
        return

    budget = args.max_chunks
    server_state: Optional[Dict[str, Any]] = None
    try:
        while True:
            manifest = load_manifest(manifest_path)
            chunk = next_pending_chunk(manifest)
            if chunk is None:
                print("no pending chunks remain in the full benchmark manifest")
                progress = write_reports(output_root, manifest_path, manifest, current_chunk_id=None, server_state=server_state)
                print_progress(progress)
                return

            if budget is not None and budget <= 0:
                print("max_chunks budget exhausted")
                progress = write_reports(output_root, manifest_path, manifest, current_chunk_id=None, server_state=server_state)
                print_progress(progress)
                return

            if server_state is None or server_state["process"].poll() is not None:
                stop_server(server_state)
                server_state = start_server(args, output_root)
                wait_for_server_ready(server_state, args.startup_timeout)

            chunk_id = chunk["chunk_id"]
            progress = write_reports(output_root, manifest_path, manifest, current_chunk_id=chunk_id, server_state=server_state)
            print_progress(progress)
            print(f"running chunk: {chunk_id}")

            return_code = run_single_chunk(args, manifest_path, chunk_id)
            if return_code != 0:
                manifest = ensure_chunk_failure_recorded(manifest_path, chunk_id, return_code)
                print(f"chunk {chunk_id} exited with return code {return_code}")
            else:
                manifest = load_manifest(manifest_path)

            progress = write_reports(output_root, manifest_path, manifest, current_chunk_id=None, server_state=server_state)
            print_progress(progress)

            if budget is not None:
                budget -= 1
    finally:
        stop_server(server_state)
        manifest = load_manifest(manifest_path) if manifest_path.exists() else {"chunks": []}
        progress = write_reports(output_root, manifest_path, manifest, current_chunk_id=None, server_state=server_state)
        print_progress(progress)


if __name__ == "__main__":
    main()
