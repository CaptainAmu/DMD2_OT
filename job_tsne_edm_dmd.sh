#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=tsne_edm_dmd_248
#SBATCH --output=logs/%j_tsne_out.txt
#SBATCH --error=logs/%j_tsne_err.txt
#SBATCH --time=6:00:00
#SBATCH --mem=36000
#SBATCH --gres=gpu:1
#SBATCH --qos=long
#SBATCH --partition=normal

export PYTHONNOUSERSITE=1

PROJECT_ROOT=/slurm-storage/shucli/PROJECT_FOLDER/DMD2_OT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p logs comparison_output/tsne_analysis

PYTHON=/slurm-storage/shucli/.conda/envs/dmd2_ot/bin/python
DMD_CKPT=model_checkpoints/pytorch_model.bin
EDM_CKPT=model_checkpoints/edm-imagenet-64x64-cond-adm.pkl

# Path to imagenet-64x64_lmdb (training data).
# Adjust this if your LMDB lives elsewhere.
IMAGENET_LMDB=model_checkpoints/imagenet-64x64_lmdb

# Target class (default: 207 = golden retriever).
CLASS_INDEX=248

# Number of shared noise seeds.
NUM_SEEDS=500

# Master RNG seed for generating the per-sample noise seeds.
MASTER_SEED=42

# EDM sampling parameters.
EDM_STEPS=256
S_CHURN=0

OUTPUT_DIR=./comparison_output/tsne_edm_vs_dmd

echo "Running t-SNE EDM vs DMD analysis..."
echo "  Class index: ${CLASS_INDEX}"
echo "  Num seeds: ${NUM_SEEDS}"
echo "  Master seed: ${MASTER_SEED}"
echo "  EDM steps: ${EDM_STEPS}, S_churn=${S_CHURN}"
echo "  LMDB path: ${IMAGENET_LMDB}"

srun -u $PYTHON -u demo/tsne_edm_dmd_imagenet.py \
    --dmd_checkpoint $DMD_CKPT \
    --edm_checkpoint $EDM_CKPT \
    --real_image_path $IMAGENET_LMDB \
    --class_index $CLASS_INDEX \
    --num_seeds $NUM_SEEDS \
    --master_seed $MASTER_SEED \
    --edm_steps $EDM_STEPS \
    --S_churn $S_CHURN \
    --output_dir "$OUTPUT_DIR"

echo "Done. Check ${OUTPUT_DIR}/${CLASS_INDEX}/t_SNE.png"

