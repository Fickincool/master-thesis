import os
from tomoSegmentPipeline.utils import setup
from cryoS2Sdrop.trainer import denoisingTrainer
from cryoS2Sdrop.dataloader import randomRotation3D


PARENT_PATH = setup.PARENT_PATH

# cet_path = os.path.join(PARENT_PATH, 'data/raw_cryo-ET/tomo02.mrc') 
cet_path = os.path.join(PARENT_PATH, 'data/S2SDenoising/dummy_tomograms/tomo02_dummy.mrc')


p=0.3 # dropout probability
subtomo_length = 128
n_features = 48

tensorboard_logdir = os.path.join(PARENT_PATH, 'data/S2SDenoising/tryout_model_logs')

batch_size = 4
epochs = 100
lr = 1e-5
num_gpus = 2

transform = randomRotation3D(0.5)

s2s_trainer = denoisingTrainer(cet_path, subtomo_length, lr, n_features, p, tensorboard_logdir)

s2s_trainer.train(batch_size, epochs, num_gpus, transform=transform)