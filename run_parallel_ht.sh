#!/bin/bash
# Run full HOI optimization pipeline on ht cluster.
# Auto-detects GPU count and distributes sequences across GPUs.
# Output goes to /mnt/afs/xinyuan/run/robust_hoi/output, not in the code dir.
#
# Usage:
#   bash run_parallel_ht.sh              # Full run, all 18 sequences
#   bash run_parallel_ht.sh --dry-run    # Quick validation: all steps, minimal params, 1 seq

set -e

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
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
source /root/envs/rhoi/bin/activate

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

# ---- Auto-detect GPUs ----
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: No GPUs detected"
  exit 1
fi
echo "Detected $NUM_GPUS GPU(s)"

# ---- Sequences ----
ALL_SEQS=(ABF12 GPMF12 GPMF14 ABF14 MC1 MDF12 MC4 MDF14 BB12 ShSu10 ShSu12 BB13 SM2 SM4 GSF12 SMu1 SMu40 GSF13)

# ---- Dry-run overrides ----
JOINT_OPT_EXTRAS=""
NEUS_EXTRAS=""
if [ "$DRY_RUN" = true ]; then
  echo "=== DRY-RUN MODE: all steps, minimal params, 1 sequence ==="
  ALL_SEQS=(SMu40)
  OUTPUT_ROOT="/mnt/afs/xinyuan/run/robust_hoi/output_dryrun"
  # joint_opt: register only 5 frames, no NeuS inside, neus_init_steps=10
  JOINT_OPT_EXTRAS="--max_register_frames 5 --no_optimize_3D_prior 1 --neus_init_steps 10"
  # neus: 10 steps
  NEUS_EXTRAS="--max_steps 10"
fi
mkdir -p "$OUTPUT_ROOT"

# ---- Run function for one GPU ----
run_on_gpu() {
  local gpu_id=$1
  shift
  local seqs="$@"
  echo "[GPU $gpu_id] Processing: $seqs"

  # Step 1: Joint optimization (pose + object mesh)
  echo "[GPU $gpu_id] Step 1/5: joint_opt"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_joint_opt \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT" \
    $JOINT_OPT_EXTRAS

  # Step 2: NeuS global (mesh reconstruction)
  echo "[GPU $gpu_id] Step 2/5: neus_global"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_neus_global \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT" \
    $NEUS_EXTRAS

  # Step 3: Hand-object alignment
  echo "[GPU $gpu_id] Step 3/5: hand alignment"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT"

  # Step 4: Evaluation
  echo "[GPU $gpu_id] Step 4/5: eval"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list hoi_pipeline_eval hoi_pipeline_eval_vis \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT"

  # Step 5: Summary
  echo "[GPU $gpu_id] Step 5/5: eval_sum"
  CUDA_VISIBLE_DEVICES=$gpu_id python run_wonder_hoi.py \
    --execute_list obj_process \
    --process_list eval_sum eval_sum_vis \
    --seq_list $seqs --rebuild --output_root "$OUTPUT_ROOT"

  echo "[GPU $gpu_id] Done: $seqs"
}

# ---- Distribute sequences across GPUs and launch ----
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
      run_on_gpu "$gpu_id" ${gpu_seqs[$gpu_id]} &
    fi
  done

  wait
fi

echo "All processes have completed successfully."
