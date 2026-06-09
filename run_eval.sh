#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name="scenecot_eval"
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=eval_output_%j.txt

source ~/miniconda3/bin/activate
conda activate scenecot

# Data + experiment roots
export SCENECOT_MODEL_ROOT=~/EE243_Scenecot_Project/model_assets
export SCENECOT_DATA_ROOT=~/EE243_Scenecot_Project/data_assets
export SCENECOT_COT_DATA_ROOT=~/EE243_Scenecot_Project/data_assets/scenecot_cot_data
export SCENECOT_EXP_ROOT=~/EE243_Scenecot_Project/experiments
export WANDB_MODE=disabled

# Model component paths
export SCENECOT_EXPERT1_PATH=$SCENECOT_MODEL_ROOT/expert1_checkpoint0
export SCENECOT_EXPERT2_PATH=$SCENECOT_MODEL_ROOT/expert2_best.pth
export SCENECOT_QUERY3D_PRETRAIN_PATH=$SCENECOT_MODEL_ROOT/query3d_pretrain/pytorch_model.bin
export SCENECOT_POINTNET_TOKENIZER_PATH=$SCENECOT_MODEL_ROOT/pointnet_tokenizer.pth
export SCENECOT_LLM_PATH=$SCENECOT_MODEL_ROOT/llava-v1.5-7b
export SCENECOT_VISION_TOWER_PATH=$SCENECOT_MODEL_ROOT/clip-vit-large-patch14-336
export SCENECOT_PQ3D_TOKENIZER_PATH=$SCENECOT_MODEL_ROOT/clip-vit-large-patch14
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

cd ~/EE243_Scenecot_Project/scenecot
sh scripts/test/full_training_msqa_beacon3d_test_moe.sh
