#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=compare_mf
#SBATCH --output=logs/%j_compare_mf_out.txt
#SBATCH --error=logs/%j_compare_mf_err.txt
#SBATCH --time=06:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=long
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs comparison_output

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python

# MeanFlow ImageNet-256 checkpoints (put teacher and student in model_checkpoints)
TEACHER_CKPT=model_checkpoints/mf_teacher_imagenet256.pkl
STUDENT_CKPT=model_checkpoints/mf_student_imagenet256.pkl

# Output subdir and image folders: edm256_images, mf_images
OUTPUT_SUBDIR=edm256_vs_mf

# ---- Control master seed here ----
MASTER_SEED=421

# ---- Number of images and teacher steps ----
NUM_IMAGES=50
TEACHER_STEPS=50

echo "Using master seed: $MASTER_SEED"
echo "Teacher: $TEACHER_CKPT ($TEACHER_STEPS steps)"
echo "Student: $STUDENT_CKPT (1 step)"
echo "Output: comparison_output/$OUTPUT_SUBDIR/"

srun -u $PYTHON -u demo/compare_teacher_meanflow.py \
    --teacher_checkpoint $TEACHER_CKPT \
    --student_checkpoint $STUDENT_CKPT \
    --output_dir ./comparison_output \
    --output_subdir $OUTPUT_SUBDIR \
    --num_images $NUM_IMAGES \
    --teacher_steps $TEACHER_STEPS \
    --master_seed $MASTER_SEED
