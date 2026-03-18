#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.libero_pro.runtime import prepare_libero_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a non-interactive LIBERO-Pro runtime for X-VLA smoke testing."
    )
    parser.add_argument(
        "--libero_pro_root",
        type=str,
        default=None,
        help="Optional path to a local LIBERO-Pro checkout. Needed for editable install.",
    )
    parser.add_argument(
        "--libero_config_path",
        type=str,
        default=None,
        help="Optional config root to use instead of ~/.libero_xvla.",
    )
    parser.add_argument(
        "--install_editable",
        action="store_true",
        help="Run `pip install -e` on --libero_pro_root in the current Python environment.",
    )
    return parser.parse_args()


def ensure_package(package_name: str) -> None:
    if importlib.util.find_spec(package_name) is not None:
        return
    subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)


def install_editable(libero_pro_root: str | None) -> None:
    if libero_pro_root is None:
        raise ValueError("--install_editable requires --libero_pro_root.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", libero_pro_root], check=True)


def verify_runtime() -> None:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    if "libero_spatial_swap" not in benchmark_dict:
        raise RuntimeError("LIBERO-Pro benchmark registry is missing libero_spatial_swap.")
    print("verified benchmark suite: libero_spatial_swap")
    print(f"verified env class: {OffScreenRenderEnv.__name__}")


def main() -> None:
    args = parse_args()

    if args.install_editable:
        install_editable(args.libero_pro_root)

    # robosuite imports termcolor at runtime, but it is not always pulled in transitively.
    ensure_package("termcolor")

    config = prepare_libero_runtime(
        libero_pro_root=args.libero_pro_root,
        config_root=args.libero_config_path,
        strict=True,
    )
    verify_runtime()

    print(f"config root: {config['config_root']}")
    print(f"config path: {config['config_path']}")
    print(f"benchmark root: {config['benchmark_root']}")


if __name__ == "__main__":
    main()
