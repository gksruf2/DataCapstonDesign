import os
from PIL import Image
import numpy as np

orig_path = '/data/gksruf293/repos/dataset/original/data'
save_path = '/data/gksruf293/repos/dataset/original'

orig_files = os.listdir(orig_path)
orig_files = [orig_path+'/'+name for name in orig_files]

# 8893 / 1000 / 2000
train_val_test_milestone = [8893, 9893, 11893]
train_val_test = ['train', 'valid', 'test']
phase = 0
split_phase = train_val_test[0]
for step, path in enumerate(orig_files):
  try:
    if step >= train_val_test_milestone[phase]:
      phase += 1
      split_phase = train_val_test[phase]
  except:
    break
  file = Image.open(path)

  print(np.asarray(file).shape)

  hr = file.resize((640,480))
  mr = file.resize((320,240))
  lr = file.resize((160,120))
  
  np_hr = np.asarray(hr)
  np_mr = np.asarray(mr)
  np_lr = np.asarray(lr)
  if step == 0:
    print(np_hr.shape)
  np.save(save_path+'/'+split_phase+'/high/'+str(step), np_hr)
  np.save(save_path+'/'+split_phase+'/mid/'+str(step), np_mr)
  np.save(save_path+'/'+split_phase+'/low/'+str(step), np_lr)