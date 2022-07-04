
import os
from tomoSegmentPipeline.utils import setup
from cryoS2Sdrop.trainer import denoisingTrainer

PARENT_PATH = setup.PARENT_PATH

cet_path = os.path.join(PARENT_PATH, 'data/raw_cryo-ET/tomo02.mrc') 

p=0.3 # dropout probability
subtomo_length = 96
n_bernoulli_samples = 35
n_features = 48

tensorboard_logdir = os.path.join(PARENT_PATH, 'data/S2SDenoising/model_logs')

batch_size = 8
epochs = 700
lr = 5e-4
num_gpus = 2

s2s_trainer = denoisingTrainer(cet_path, subtomo_length, lr, n_bernoulli_samples, n_features, p, tensorboard_logdir)

s2s_trainer.train(batch_size, epochs, num_gpus)