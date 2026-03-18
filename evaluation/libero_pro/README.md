# LIBERO-Pro Evaluation

This directory contains the minimum chunked integration needed to run the `2toINF/X-VLA-Libero` checkpoint on the LIBERO-Pro benchmark.

The current phases are:
- `smoke`: one chunk, `libero_spatial x task 0 x position perturbation x seed 42`
- `mini`: 32 chunks, `4 suites x first 2 tasks x {position, semantic} x {42, 43}`
- `full`: 80 chunks, `4 suites x 10 tasks x {position, semantic} x seed 42`, with `50` rollouts per chunk

For LIBERO-Pro, the position perturbation maps to:
- `use_swap: true` in `evaluation_config.yaml`
- benchmark suite suffix `_swap`, so the runtime suite is `libero_spatial_swap`

For the semantic perturbation used in the mini benchmark:
- benchmark suite suffix `_lan`

## 1. Prepare LIBERO-Pro

Install LIBERO-Pro in a separate environment that follows LIBERO's dependency versions.

If you already have a local LIBERO-Pro checkout, run the helper below inside that environment first. It avoids the package's interactive `~/.libero/config.yaml` prompt and writes a dedicated config under `~/.libero_xvla` instead.

```bash
cd /home/nhattm1/vinhtt19/X-VLA
python evaluation/libero_pro/setup_runtime.py \
  --libero_pro_root /path/to/LIBERO-PRO \
  --install_editable
```

This helper also installs `termcolor` if it is missing, since `robosuite` imports it at runtime.

Download the LIBERO-Pro `bddl_files` and `init_files`, then place them under the installed LIBERO-Pro tree:

```bash
mv libero_data/bddl_files/* libero/libero/bddl_files/
mv libero_data/init_files/* libero/libero/init_files/
```

## 2. Start the X-VLA server

```bash
cd /home/nhattm1/vinhtt19/X-VLA
python -m deploy \
  --model_path /path/to/X-VLA-Libero \
  --port 8000 \
  --output_dir logs/libero_pro_server
```

## 3. Build a manifest

Run this inside the LIBERO-Pro-compatible environment:

```bash
cd /home/nhattm1/vinhtt19/X-VLA
python evaluation/libero_pro/build_manifest.py \
  --phase smoke \
  --output_root logs/libero_pro_smoke \
  --libero_pro_root /path/to/LIBERO-PRO
```

For the mini benchmark, use:

```bash
cd /home/nhattm1/vinhtt19/X-VLA
python evaluation/libero_pro/build_manifest.py \
  --phase mini \
  --output_root logs/libero_pro_mini \
  --libero_pro_root /path/to/LIBERO-PRO
```

This creates:
- `smoke`: a single pending chunk
- `mini`: a 32-chunk queue in deterministic order
- `full`: an 80-chunk queue for the full validated LIBERO-Pro scope

For the full benchmark, use:

```bash
cd /home/nhattm1/vinhtt19/X-VLA
python evaluation/libero_pro/build_manifest.py \
  --phase full \
  --output_root logs/libero_pro_full \
  --libero_pro_root /path/to/LIBERO-PRO
```

## 4. Run the chunk

```bash
cd /home/nhattm1/vinhtt19/X-VLA
python evaluation/libero_pro/run_chunk.py \
  --manifest_path logs/libero_pro_smoke/smoke_manifest.json \
  --run_next \
  --connection_info logs/libero_pro_server/info.json \
  --libero_pro_root /path/to/LIBERO-PRO \
  --steps 10 \
  --save_videos \
  --fail_fast
```

For mini-benchmark chunks, point `--manifest_path` at `logs/libero_pro_mini/mini_manifest.json`. Keep the execution model chunked:

```bash
cd /home/nhattm1/vinhtt19/X-VLA
python evaluation/libero_pro/run_chunk.py \
  --manifest_path logs/libero_pro_mini/mini_manifest.json \
  --run_next \
  --server_ip 127.0.0.1 \
  --server_port 8011 \
  --libero_pro_root /path/to/LIBERO-PRO \
  --steps 10 \
  --video_policy all
```

Each invocation runs exactly one chunk and exits.

The chunk runner now supports:
- `--video_policy none`
- `--video_policy all`
- `--video_policy failures_only`

## 5. Run the full benchmark

Use the managed full-benchmark runner inside the `libero_pro` environment:

```bash
cd /home/nhattm1/vinhtt19/X-VLA
python evaluation/libero_pro/run_full_benchmark.py \
  --model_path /path/to/X-VLA-Libero \
  --output_root logs/libero_pro_full \
  --libero_pro_root /path/to/LIBERO-PRO \
  --video_policy failures_only
```

Useful options:
- `--cleanup_legacy_logs` deletes `logs/libero_pro_smoke` and `logs/libero_pro_mini` after ETA priors are captured
- `--retry_failed` resets failed chunks to pending before resuming
- `--max_chunks 1` runs a single full-benchmark chunk for validation
- `--max_chunks 0` prepares the manifest and report files without launching any chunk

The managed runner:
- creates or resumes `full_manifest.json`
- starts `deploy.py` on GPU and waits for readiness
- runs chunks sequentially
- updates `progress.json`, `results_by_chunk.csv`, `results_by_suite.csv`, `results_by_task.csv`, and `summary.md` after every chunk

## 6. Inspect outputs

After the run, inspect:
- the manifest JSON for `status`, `error`, `started_at`, `finished_at`, `success_rate`, `attempt_count`, and `last_duration_seconds`
- the chunk output directory under `logs/libero_pro_smoke/chunks/...`, `logs/libero_pro_mini/chunks/...`, or `logs/libero_pro_full/chunks/...`
- `logs/libero_pro_full/progress.json` for current counts and ETA
- `logs/libero_pro_full/summary.md` for a human-readable checkpoint
- the CSV reports under `logs/libero_pro_full/`

Expected chunk artifacts:
- `rollout.json`
- `summary.json`
- `rollout_ep0.mp4` when the selected video policy retains that episode
- `error.txt` if the rollout fails
