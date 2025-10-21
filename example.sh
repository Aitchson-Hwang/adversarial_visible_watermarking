#!/bin/bash
# ================================================================
# Script Name: run_adversarial_watermark_removal.sh
# Description:
#   This script runs the Adversarial Visible Watermarking experiment
#   using the SLBR model (DENet, SplitNet, MNet, etc. are also applicable. 
#        You only need to (or not) add some fields unique to their models. 
#        You can find them in the .sh files in their source code.). 
#   This script launches the training or attack process
#   with customizable parameters such as learning rate, dataset path,
#   and attack configuration.
#
# Usage:
#   bash example.sh
#
# Notes:
#   - Modify the paths below to match your environment.
#   - Requires a CUDA-capable GPU and dependencies from requirements.txt.
#
# Arguments:
#   --epochs             Number of training epochs.
#   --schedule           Learning rate schedule strategy.
#   --lr                 Initial learning rate.
#   --resume             Path to the pretrained watermark removal model checkpoint.
#   --arch               Model architecture to use (e.g., SLBR).
#   -c                   Path to save.
#   --attack_method      Adversarial attack method (e.g., pgd_inn, pgd).
#   --limited-dataset    Whether to use a limited dataset (1 = True).
#   --use_rie            Enable the RIE module to insert perturabtion.
#   --epsilon            Perturbation limit for general adversarial attack (e.g., pgd).
#   --step_alpha         Step size for PGD attack.
#   --stopnum            How many adversairal images to generate.
#   --iters              Number of general attack iterations.
#   --rie_iters          Number of the RIE module iterations.
#   --simage             Whether to save adversarial images.
#   --lambda_p           Loss weight for perceptual loss.
#   --input-size         Input image size.
#   --train-batch        Training batch size.
#   --test-batch         Testing batch size.
#   --use_refine         Enable refinement module, only used for the SLBR model.
#   --base-dir           Path to the dataset root directory.
#   --data               Dataset name (e.g., 'multi', 'full', '10kgray', '10kmid', etc.).
#
# Example:
#   bash example.sh
# ================================================================

set -ex
 CUDA_VISIBLE_DEVICES=0 python adversarial_visible_watermarking/main.py  --epochs 1\
 --schedule 1\
 --lr 1e-4\
 --resume adversarial_visible_watermarking/wm_removers/ckpt/SLBR.pth.tar\
 --arch slbr\
 -c /path/to/your/save_path\
 --attack_method pgd_inn\
 --limited-dataset 1\
 --use_rie True\
 --epsilon 4\
 --step_alpha 1\
 --stopnum 5\
 --iters 50\
 --rie_iters 200\
 --simage True\
 --lambda_p 200\
 --input-size 256\
 --train-batch 1\
 --test-batch 1\
 --use_refine \
 --base-dir /path/to/your/dataset\
 --data dataset_name
