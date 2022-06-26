#!/bin/bash

#SBATCH -o /home/haicu/jeronimo.carvajal/logs_slurm/slurm_jupyter_storage_%j.job
#SBATCH -e /home/haicu/jeronimo.carvajal/logs_slurm/slurm_jupyter_storage_%j.job

#SBATCH -J isoNet
#SBATCH -c 6 #CPU cores required
#SBATCH --gres=gpu:3

#SBATCH --mem=24G #Memory required
#SBATCH -t 1-20:00:00 #Job runtime
#SBATCH --partition=gpu_p
#SBATCH --qos gpu
##SBATCH --nodelist=supergpu03pxe

## SBATCH --mem=16G #Memory required
## SBATCH -t 12:00:00 #Job runtime
## SBATCH --partition=interactive_gpu_p
## SBATCH --qos interactive_gpu

#SBATCH --mail-user=jeronimo.carvajal@helmholtz-munich.de

#SBATCH --nice=10000 #Manual priority. Do not change this.

source $HOME/.bashrc
# chmod 600 $HOME/logs_slurm/slurm_jupyter_storage_$SLURM_JOB_ID.job

# do stuff
conda deactivate
conda activate isonet

########################################### CRYOCARE DATASET ###############################
# isonet.py prepare_star cryoCARE_tomoset --output_star cryoCARE_tomos.star --pixel_size 14.08 # first run 10
# isonet.py make_mask cryoCARE_tomos.star --mask_folder cryoCARE_tomos_mask --density_percentage 50 --std_percentage 50
# isonet.py extract cryoCARE_tomos.star --subtomo_folder cryoCARE_subtomo --subtomo_star cryoCARE_subtomo.star
# isonet.py refine cryoCARE_subtomo.star --gpuID 0,1,2 --iterations 30 --result_dir cryoCARE_results --noise_level 0.0 --noise_start_iter 31
# # isonet.py predict cryoCARE_tomos.star cryoCARE_results/model_iter30.h5 --gpuID 0,1,2 --output_dir cryoCARE_corrected

# # predict the rest of the tomograms
# isonet.py prepare_star cryoCARE_completeTomoset --output_star cryoCARE_allTomos.star --pixel_size 14.08 # first run 10
# isonet.py predict cryoCARE_allTomos.star cryoCARE_results/model_iter30.h5 --gpuID 0,1,2 --output_dir cryoCARE_corrected


########################################### RAW TOMOGRAMS ######################################
# cd $HOME/Thesis/data/isoNet/RAW_dataset/
# training job 6247245
# isonet.py prepare_star RAW_tomoset --output_star RAW_tomos.star --pixel_size 14.08
# isonet.py deconv RAW_tomos.star --snrfalloff 1 --deconvstrength 1 --deconv_folder RAW_tomos_deconv
# isonet.py make_mask RAW_tomos.star --mask_folder RAW_tomos_mask --density_percentage 50 --std_percentage 50 --z_crop 0.25
# isonet.py extract RAW_tomos.star --subtomo_folder RAW_subtomo --subtomo_star RAW_subtomo.star --cube_size 96

# Iterate
# isonet.py refine RAW_subtomo_sanityChecks.star --gpuID 0,1,2 --iterations 20 --result_dir refine_pilot/run1/RAW_results --noise_level 0.05,0.1 --noise_start_iter 15,20 --batch_size 12 --learning_rate 0.001

# predict the rest of the tomograms
# isonet.py prepare_star RAW_completeTomoset --output_star RAW_allTomos.star --pixel_size 14.08
# isonet.py deconv RAW_allTomos.star --snrfalloff 1 --deconvstrength 1 --deconv_folder RAW_allTomos_deconv
# isonet.py predict RAW_allTomos.star refine/runAAAA/RAW_results/model_iter30.h5 --gpuID 0,1,2 --output_dir refine/runAAA/RAW_corrected_i30


########################################### SALMONELLA TOMOGRAMS ######################################
cd $HOME/Thesis/data/isoNet/salmonella_dataset/
isonet.py prepare_star salmonella_minicells --output_star tomos.star --pixel_size 13.6
isonet.py deconv tomos.star --snrfalloff 1 --deconvstrength 1 --deconv_folder tomos_deconv
isonet.py make_mask tomos.star --mask_folder tomos_mask --density_percentage 50 --std_percentage 50 --z_crop 0.3
isonet.py extract tomos.star --subtomo_folder subtomo --subtomo_star subtomo.star --cube_size 96

isonet.py refine subtomo.star --gpuID 0,1,2 --iterations 30 --result_dir results --noise_level 0.05,0.1 --noise_start_iter 20,25
isonet.py predict tomos.star results/model_iter30.h5 --gpuID 0,1,2 --output_dir corrected_i30

## Try running this with default values to check everything works
# isonet.py prepare_star cryoCARE_tomoset --output_star nnUnet_tomos_denoised.star --pixel_size 10
# isonet.py deconv nnUnet_tomos_denoised.star --snrfalloff 0.5 --deconv_folder tomos_deconv
# isonet.py make_mask nnUnet_tomos_denoised.star --mask_folder tomos_mask --density_percentage 50 --std_percentage 50
# isonet.py extract nnUnet_tomos_denoised.star
# isonet.py refine subtomo.star --gpuID 0,1,2 --iterations 30
# isonet.py predict nnUnet_tomos_denoised.star results/model_iter30.h5 --gpuID 0,1,2