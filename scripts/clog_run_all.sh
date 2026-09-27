#!/usr/bin/env bash
# Run the whole pipe-clog pipeline end to end (Git Bash on Windows). Takes about 10 minutes on a laptop CPU.
#   bash scripts/clog_run_all.sh > data/raw/clog/run_all.log 2>&1
# RT = runtime env (origin_hack), ML = simulation env (cerebro_ml: wntr 1.5.0, pyswmm 2.1.0).
set -euo pipefail
cd "$(dirname "$0")/.."
RT=${RT:-/c/Users/HP/miniconda3/envs/origin_hack/python.exe}
ML=${ML:-/e/conda_envs/cerebro_ml/python.exe}
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 TMP='E:\tmp' TEMP='E:\tmp' HF_HOME='E:\hf_cache' \
       TORCH_HOME='E:\torch_cache' PIP_CACHE_DIR='E:\pip_cache'
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
step "1 fetch + sha256 manifest";      "$RT" scripts/fetch_clog_data.py --no-net
step "2 riser simulation (WNTR)";      "$ML" scripts/clog_riser_sim.py --jobs 3
step "3 riser detectors + eval";       "$RT" scripts/clog_riser_eval.py --boot 1000
step "4 drain-down simulation (SWMM)"; "$ML" scripts/clog_drain_sim.py
step "5 Bellinge real blockage";       "$RT" scripts/clog_bellinge.py
step "6 check schedule";               "$RT" scripts/clog_schedule.py --as-of 2026-09-26
step "done"
