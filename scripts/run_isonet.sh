#!/bin/bash
# conda deactivate
# conda activate isonet

############# Lung Dummy dataset ####################################################################3
cd $HOME/Thesis/data/isoNet/lung_dummyData_ae35/
isonet.py prepare_star lung_corrupted --output_star tomos.star 
isonet.py make_mask tomos.star --mask_folder tomos_mask --density_percentage 80 --std_percentage 50
isonet.py extract tomos.star --subtomo_folder subtomo --subtomo_star subtomo.star --cube_size 96

isonet.py refine subtomo.star --gpuID 0,1 --iterations 30 --result_dir results --noise_level 0.00 --noise_start_iter 29 
# isonet.py refine subtomo.star --continue_from results/refine_iter19.json --gpuID 0,1 --iterations 30 --result_dir results --noise_level 0.00 --noise_start_iter 29 
isonet.py predict tomos.star results/model_iter30.h5 --gpuID 0,1 --output_dir corrected_i30


############# Brain Dummy dataset ####################################################################3
# cd $HOME/Thesis/data/isoNet/brain_dummyData_ae15/
# isonet.py prepare_star brain_corrupted --output_star tomos.star 
# isonet.py make_mask tomos.star --mask_folder tomos_mask --density_percentage 80 --std_percentage 50
# isonet.py extract tomos.star --subtomo_folder subtomo --subtomo_star subtomo.star --cube_size 80

# isonet.py refine subtomo.star --gpuID 0,1 --iterations 30 --result_dir results --noise_level 0.00 --noise_start_iter 29
# isonet.py predict tomos.star results/model_iter30.h5 --gpuID 0,1 --output_dir corrected_i30

############ Brain Dummy dataset ####################################################################3
# cd $HOME/Thesis/data/isoNet/brain_dummyData_rotated/
# isonet.py prepare_star brain_corrupted --output_star tomos.star 
# isonet.py make_mask tomos.star --mask_folder tomos_mask --density_percentage 80 --std_percentage 50
# isonet.py extract tomos.star --subtomo_folder subtomo --subtomo_star subtomo.star --cube_size 80

# isonet.py refine subtomo.star --gpuID 0,1 --iterations 30 --result_dir results --noise_level 0.00 --noise_start_iter 29
# isonet.py predict tomos.star results/model_iter30.h5 --gpuID 0,1 --output_dir corrected_i30

rm -rf results/data/