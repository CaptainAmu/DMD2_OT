#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=distill_cifar10
#SBATCH --output=logs/%j_distill_cifar10_out.txt
#SBATCH --error=logs/%j_distill_cifar10_err.txt
#SBATCH --time=1-00:00:00
#SBATCH --mem=48000
#SBATCH --gres=gpu:1
#SBATCH --qos=medium
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs cifar10_distill_runs

# ======= EDIT THESE BEFORE SUBMITTING =======
WANDB_ENTITY="shucheng_li-university-of-oxford"          # your wandb username or team
WANDB_PROJECT="DMD2_OT"  # project name
RUN_NAME="distill_cifar10"

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python
TEACHER_PATH=model_checkpoints/edm-cifar10-32x32-cond-vp.pkl
FID_REF_PATH=model_checkpoints/cifar10-32x32_fid_ref.npz
OUTPUT_DIR="${PROJECT_ROOT}/cifar10_distill_runs/${RUN_NAME}"

srun -u $PYTHON -m demo.cifar10_distill \
    --teacher_path "$TEACHER_PATH" \
    --fid_ref_path "$FID_REF_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 128 \
    --num_epochs 50 \
    --steps_per_epoch 1000 \
    --lr_gen 2e-4 \
    --lr_score 2e-4 \
    --fid_num_samples_epoch 5000 \
    --fid_num_samples_step 500 \
    --fid_batch_size 128 \
    --log_interval_steps 200 \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME"

