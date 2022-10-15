#!/bin/bash
# conda deactivate
# conda activate isonet

########################################### RAW TOMOGRAMS ######################################
cd $HOME/Thesis/data/isoNet/RAW_dataset/
isonet.py prepare_star RAW_tomoset --output_star RAW_tomos.star --pixel_size 14.08 --defocus -0.25
isonet.py deconv RAW_tomos.star --snrfalloff 0.6 --deconvstrength 0.45 --highpassnyquist 0.02 --deconv_folder RAW_tomos_deconv
isonet.py make_mask RAW_tomos.star --mask_folder RAW_tomos_mask --density_percentage 80 --std_percentage 50 --z_crop 0.25
isonet.py extract RAW_tomos.star --subtomo_folder RAW_subtomo --subtomo_star RAW_subtomo.star --cube_size 96

# Iterate
# isonet.py refine RAW_subtomo_sanityChecks.star --gpuID 0,1,2 --iterations 20 --result_dir refine_pilot/run1/RAW_results --noise_level 0.05,0.1 --noise_start_iter 15,20 --batch_size 12 --learning_rate 0.001

# predict the rest of the tomograms
# isonet.py prepare_star RAW_completeTomoset --output_star RAW_allTomos.star --pixel_size 14.08
# isonet.py deconv RAW_allTomos.star --snrfalloff 1 --deconvstrength 1 --deconv_folder RAW_allTomos_deconv
# isonet.py predict RAW_allTomos.star refine/runAAAA/RAW_results/model_iter30.h5 --gpuID 0,1,2 --output_dir refine/runAAA/RAW_corrected_i30


########################################### CRYOCARE DATASET ###############################
# cd $HOME/Thesis/data/isoNet/cryoCARE_dataset/
# isonet.py prepare_star cryoCARE_tomoset --output_star cryoCARE_tomos.star --pixel_size 14.08 
# isonet.py make_mask cryoCARE_tomos.star --mask_folder cryoCARE_tomos_mask --density_percentage 80 --std_percentage 50 --z_crop 0.25
# isonet.py extract cryoCARE_tomos.star --subtomo_folder cryoCARE_subtomo --subtomo_star cryoCARE_subtomo.star
# isonet.py refine cryoCARE_subtomo.star --gpuID 0,1,2 --iterations 30 --result_dir cryoCARE_results --noise_level 0.0 --noise_start_iter 31
# # isonet.py predict cryoCARE_tomos.star cryoCARE_results/model_iter30.h5 --gpuID 0,1,2 --output_dir cryoCARE_corrected

# # predict the rest of the tomograms
# isonet.py prepare_star cryoCARE_completeTomoset --output_star cryoCARE_allTomos.star --pixel_size 14.08
# isonet.py predict cryoCARE_allTomos.star cryoCARE_results/model_iter30.h5 --gpuID 0,1,2 --output_dir cryoCARE_corrected

rm -rf results/data/