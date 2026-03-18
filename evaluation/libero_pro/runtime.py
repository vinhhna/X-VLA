#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Optional


DEFAULT_CONFIG_ROOT = Path.home() / ".libero_xvla"


def _normalize_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value is None:
        return None
    return Path(path_value).expanduser().resolve()


def _add_repo_root_to_path(libero_pro_root: Path) -> None:
    root_str = str(libero_pro_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _resolve_benchmark_root(candidate_root: Path) -> Optional[Path]:
    candidate_root = candidate_root.resolve()
    inner_root = candidate_root / "libero" / "libero"
    if inner_root.exists():
        return inner_root

    if (candidate_root / "bddl_files").exists() and (candidate_root / "init_files").exists():
        return candidate_root

    return None


def _infer_benchmark_root_from_installed_package() -> Optional[Path]:
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        return None
    package_root = Path(spec.origin).resolve().parent
    return _resolve_benchmark_root(package_root)


def infer_benchmark_root(libero_pro_root: Optional[str] = None) -> Optional[Path]:
    explicit_root = _normalize_path(libero_pro_root)
    if explicit_root is not None:
        _add_repo_root_to_path(explicit_root)
        benchmark_root = _resolve_benchmark_root(explicit_root)
        if benchmark_root is None:
            raise FileNotFoundError(
                f"Could not find LIBERO-Pro benchmark files under {explicit_root}. "
                "Expected either <root>/libero/libero or a direct benchmark root."
            )
        return benchmark_root

    return _infer_benchmark_root_from_installed_package()


def build_config_dict(benchmark_root: Path) -> Dict[str, str]:
    benchmark_root = benchmark_root.resolve()
    return {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str((benchmark_root / "bddl_files").resolve()),
        "init_states": str((benchmark_root / "init_files").resolve()),
        "datasets": str((benchmark_root.parent / "datasets").resolve()),
        "assets": str((benchmark_root / "assets").resolve()),
    }


def write_config(config_root: Path, config: Dict[str, str]) -> Path:
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"

    lines = [f"{key}: {value}\n" for key, value in config.items()]
    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return config_path


def prepare_libero_runtime(
    *,
    libero_pro_root: Optional[str] = None,
    config_root: Optional[str] = None,
    strict: bool = True,
) -> Optional[Dict[str, str]]:
    benchmark_root = infer_benchmark_root(libero_pro_root=libero_pro_root)
    if benchmark_root is None:
        if strict:
            raise ModuleNotFoundError(
                "LIBERO-Pro runtime is not importable. Install it in the active Python "
                "environment or provide --libero_pro_root pointing to a local checkout."
            )
        return None

    config = build_config_dict(benchmark_root)
    resolved_config_root = _normalize_path(config_root) or DEFAULT_CONFIG_ROOT
    config_path = write_config(resolved_config_root, config)
    os.environ["LIBERO_CONFIG_PATH"] = str(resolved_config_root)

    for path_value in config.values():
        path = Path(path_value)
        if path.name == "datasets" and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            message = f"Required LIBERO-Pro path does not exist: {path}"
            if strict:
                raise FileNotFoundError(message)
            print(f"[warning] {message}")

    config["config_root"] = str(resolved_config_root)
    config["config_path"] = str(config_path)
    return config
