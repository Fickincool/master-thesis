import os 

configuration = '3d_fullres'
trainer_class_name = 'nnUNetTrainerV2'
task_name = 'Task778_rawCETBaseline'


for fold in range(5):
    os.system(f'nnUNet_train {configuration} {trainer_class_name} {task_name} {fold} --npz')
