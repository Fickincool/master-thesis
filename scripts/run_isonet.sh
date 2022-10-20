#!/bin/bash
# conda deactivate
# conda activate isonet

########################################### RAW TOMOGRAMS ######################################
# cd $HOME/Thesis/data/isoNet/RAW_dataset/
# isonet.py prepare_star RAW_tomoset --output_star RAW_tomos.star --pixel_size 14.08 --defocus -0.25
# isonet.py deconv RAW_tomos.star --snrfalloff 0.6 --deconvstrength 0.45 --highpassnyquist 0.02 --deconv_folder RAW_tomos_deconv
# isonet.py make_mask RAW_tomos.star --mask_folder RAW_tomos_mask --density_percentage 80 --std_percentage 50 --z_crop 0.25
# isonet.py extract RAW_tomos.star --subtomo_folder RAW_subtomo --subtomo_star RAW_subtomo.star --cube_size 96

# Iterate
# isonet.py refine RAW_subtomo.star --gpuID 0,1 --iterations 30 --result_dir RAW_results --learning_rate 0.0005
# rm -rf RAW_results/data/

# predict the rest of the tomograms
# isonet.py prepare_star RAW_allTomos --output_star RAW_allTomos.star --pixel_size 14.08 --defocus -0.25
# isonet.py deconv RAW_allTomos.star --snrfalloff 0.6 --deconvstrength 0.45 --highpassnyquist 0.02 --deconv_folder RAW_allTomos_deconv
# isonet.py predict RAW_allTomos.star --model RAW_results/model_iter30.h5 --gpuID 0,1 --output_dir RAW_corrected



########################################### SHREC TOMOGRAMS ######################################
# cd $HOME/Thesis/data/isoNet/SHREC_dataset/
# isonet.py prepare_star SHREC_tomoset --output_star SHREC_tomos.star --pixel_size 10 --defocus 0
# isonet.py deconv SHREC_tomos.star --snrfalloff 0.3 --deconvstrength 1 --highpassnyquist 0.02 --deconv_folder SHREC_tomos_deconv
# isonet.py make_mask SHREC_tomos.star --mask_folder SHREC_tomos_mask --density_percentage 80 --std_percentage 50
# isonet.py extract SHREC_tomos.star --subtomo_folder SHREC_subtomo --subtomo_star SHREC_subtomo.star --cube_size 96

# Iterate
# isonet.py refine SHREC_subtomo.star --gpuID 0,1 --iterations 30 --result_dir SHREC_results --learning_rate 0.0005
# rm -rf SHREC_results/data/

# isonet.py predict SHREC_tomos.star --model SHREC_results/model_iter30.h5 --gpuID 0,1 --output_dir SHREC_corrected

########################################### CRYOCARE DATASET ###############################
# cd $HOME/Thesis/data/isoNet/cryoCARE_dataset/
# isonet.py prepare_star cryoCARE_tomoset --output_star cryoCARE_tomos.star --pixel_size 14.08 --defocus -0.25
# isonet.py make_mask cryoCARE_tomos.star --mask_folder cryoCARE_tomos_mask --density_percentage 80 --std_percentage 50 --z_crop 0.25
# isonet.py extract cryoCARE_tomos.star --subtomo_folder cryoCARE_subtomo --subtomo_star cryoCARE_subtomo.star
# isonet.py refine cryoCARE_subtomo.star --gpuID 0,1,2 --iterations 30 --result_dir cryoCARE_results --noise_level 0.0 --noise_start_iter 31
# # isonet.py predict cryoCARE_tomos.star cryoCARE_results/model_iter30.h5 --gpuID 0,1,2 --output_dir cryoCARE_corrected

# # predict the rest of the tomograms
# isonet.py prepare_star cryoCARE_completeTomoset --output_star cryoCARE_allTomos.star --pixel_size 14.08
# isonet.py predict cryoCARE_allTomos.star cryoCARE_results/model_iter30.h5 --gpuID 0,1,2 --output_dir cryoCARE_corrected