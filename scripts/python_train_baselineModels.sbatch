#!/bin/bash

#SBATCH -o /home/haicu/jeronimo.carvajal/logs_slurm/slurm_jupyter_storage_%j.job
#SBATCH -e /home/haicu/jeronimo.carvajal/logs_slurm/slurm_jupyter_storage_%j.job

#SBATCH -J trainBaseline
#SBATCH -c 6 #CPU cores required
#SBATCH --gres=gpu:3

#SBATCH --mem=24G #Memory required
#SBATCH -t 2-00:00:00 #Job runtime
#SBATCH --partition=gpu_p
#SBATCH --qos gpu
#SBATCH --nodelist=supergpu07

##SBATCH --mem=16G #Memory required
##SBATCH -t 12:00:00 #Job runtime
##SBATCH --partition=interactive_gpu_p
##SBATCH --qos interactive_gpu

#SBATCH --mail-user=jeronimo.carvajal@helmholtz-munich.de

#SBATCH --nice=10000 #Manual priority. Do not change this.

source $HOME/.bashrc

# do stuff
conda deactivate
conda activate jero_supergpu

cd $HOME/Thesis/scripts/
# launch the jupyter instance (this works with jupyter notebook as well) 
python train_baselineModels.py