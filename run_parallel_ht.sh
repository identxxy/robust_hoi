#!/bin/bash
# Run full HOI optimization pipeline on ht cluster.
# Auto-detects GPU count and distributes sequences across GPUs.
# Output goes to /mnt/afs/xinyuan/run/robust_hoi/output, not in the code dir.
#
# Usage:
#   bash run_parallel_ht.sh              # Full run, all 18 sequences
#   bash run_parallel_ht.sh --dry-run    # Quick validation: all steps, minimal params, 1 seq

set -e

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- Parse args ----
DRY_RUN=false
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
  esac
done

# ---- Environment ----
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
source /root/envs/rhoi/bin/activate
# Put PyTorch's bundled CUDA libs first so they take precedence over system CUDA libs.
# (System /usr/local/cuda/lib64 may have older libnvJitLink that breaks torch import.)
# Collect pip-installed nvidia CUDA lib dirs (e.g. nvidia-nvjitlink-cu12, nvidia-cusparse-cu12, ...)
# These must come BEFORE system /usr/local/cuda/lib64 to avoid stale symbol errors.
# Search both venv site-packages and system dist-packages.
NVIDIA_LIBS=$(python -c "
import glob, os, sys
patterns = [
    os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'lib/python*/site-packages/nvidia/*/lib'),
    '/usr/local/lib/python*/dist-packages/nvidia/*/lib',
]
dirs = []
for p in patterns:
    dirs.extend(glob.glob(p))
print(':'.join(d for d in dirs if os.path.isdir(d)))
" 2>/dev/null || echo "")
if [ -n "$NVIDIA_LIBS" ]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIBS:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
  log "nvidia pip libs prepended to LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
  log "WARNING: no pip nvidia libs found, using system CUDA libs only"
fi
log "CUDA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
log "/root mount: $(df -h /root | tail -1)"
log "/root/envs/rhoi exists: $(ls /root/envs/rhoi/bin/python 2>/dev/null && echo YES || echo NO)"
TORCH_VER=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "FAILED")
log "Python: $(which python)  torch: $TORCH_VER"
if [ "$TORCH_VER" = "FAILED" ]; then
  log "ERROR: torch import failed. Diagnostics:"
  log "  /root mount: $(mount | grep '/root' || echo 'not a mountpoint')"
  log "  torch location: $(find /root/envs/rhoi -name 'torch' -maxdepth 6 -type d 2>/dev/null | head -3)"
  python -c 'import torch' 2>&1 | tail -5 >&2
  exit 1
fi

export DATASET=ho3d
export DATASET_DIR="/mnt/afs/xinyuan/data/HOD3D_v3/train/"
export CONDA_DIR="/root"
export RHOI_ENV="rhoi"
export BODY_MODELS_DIR="/mnt/afs/xinyuan/data/body_models"
export GT_PROCESSED_DIR="/mnt/afs/xinyuan/data/HOD3D_v3/processed"
export OUTPUT_BASELINE_DIR="/mnt/afs/xinyuan/run/robust_hoi/output_baseline"

CODE_DIR="/mnt/afs/xinyuan/code/robust_hoi"
OUTPUT_ROOT="/mnt/afs/xinyuan/run/robust_hoi/output"

cd "$CODE_DIR"
log "CODE_DIR=$CODE_DIR"

# ---- One-time fixes for known environment issues ----
# FoundationPose's mycpp extension needs __init__.py to be importable as mycpp.build.mycpp
touch third_party/FoundationPose/mycpp/__init__.py
touch third_party/FoundationPose/mycpp/build/__init__.py

# ---- Auto-detect GPUs ----
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")
if [ "$NUM_GPUS" -eq 0 ]; then
  log "ERROR: No GPUs detected"
  exit 1
fi
log "Detected $NUM_GPUS GPU(s):"
nvidia-smi -L | sed 's/^/  /'

# ---- Sequences ----
ALL_SEQS=(ABF12 GPMF12 GPMF14 ABF14 MC1 MDF12 MC4 MDF14 BB12 ShSu10 ShSu12 BB13 SM2 SM4 GSF12 SMu1 SMu40 GSF13)

# ---- Dry-run overrides ----
JOINT_OPT_EXTRAS=""
NEUS_EXTRAS=""
if [ "$DRY_RUN" = true ]; then
  log "=== DRY-RUN MODE: all steps, minimal params, 1 sequence ==="
  ALL_SEQS=(SMu40)
  OUTPUT_ROOT="/mnt/afs/xinyuan/run/robust_hoi/output_dryrun"
  # joint_opt: register only 5 frames, no NeuS inside, neus_init_steps=10
  JOINT_OPT_EXTRAS="--max_register_frames 5 --no_optimize_3D_prior 1 --neus_init_steps 10"
  # neus: 10 steps
  NEUS_EXTRAS="--max_steps 10"
fi
log "OUTPUT_ROOT=$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# ---- Run function for one GPU ----
run_on_gpu() {
  local gpu_id=$1
  shift
  local seqs="$@"
  local n_seqs
  n_seqs=$(echo $seqs | wc -w)
  log "[GPU $gpu_id] Starting $n_seqs sequences: $seqs"

  # Step 1: Joint optimization (pose + object mesh)
  log "[GPU $gpu_id] Step 1/5 start: joint_opt"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_joint_opt \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT" \
    $JOINT_OPT_EXTRAS
  log "[GPU $gpu_id] Step 1/5 done:  joint_opt"

  # Step 2: NeuS global (mesh reconstruction)
  log "[GPU $gpu_id] Step 2/5 start: neus_global"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_neus_global \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT" \
    $NEUS_EXTRAS
  log "[GPU $gpu_id] Step 2/5 done:  neus_global"

  # Step 3: Hand-object alignment
  log "[GPU $gpu_id] Step 3/5 start: hand alignment"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT"
  log "[GPU $gpu_id] Step 3/5 done:  hand alignment"

  # Step 4: Evaluation
  log "[GPU $gpu_id] Step 4/5 start: eval"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_eval hoi_pipeline_eval_vis \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT"
  log "[GPU $gpu_id] Step 4/5 done:  eval"

  # Step 5: Summary
  log "[GPU $gpu_id] Step 5/5 start: eval_sum"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list eval_sum eval_sum_vis \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT"
  log "[GPU $gpu_id] Step 5/5 done:  eval_sum"

  log "[GPU $gpu_id] ALL DONE: $seqs"
}

# ---- Distribute sequences across GPUs and launch ----
log "Distributing ${#ALL_SEQS[@]} sequences across $NUM_GPUS GPU(s)..."
if [ "$NUM_GPUS" -eq 1 ]; then
  run_on_gpu 0 "${ALL_SEQS[@]}"
else
  declare -A gpu_seqs
  for i in "${!ALL_SEQS[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    gpu_seqs[$gpu_id]="${gpu_seqs[$gpu_id]} ${ALL_SEQS[$i]}"
  done

  for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -n "${gpu_seqs[$gpu_id]}" ]; then
      log "[GPU $gpu_id] Assigned:$(echo ${gpu_seqs[$gpu_id]} | wc -w) seqs -${gpu_seqs[$gpu_id]}"
      run_on_gpu "$gpu_id" ${gpu_seqs[$gpu_id]} &
    fi
  done

  wait
fi

log "All processes completed successfully."
