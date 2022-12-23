import cv2
import os
import numpy as np
import natsort
from torchvision import datasets, io, transforms
from PIL import Image

def jpg_to_numpy(orig_path, hr_path, mr_path, lr_path):
    r"""
    Make hr, mr, lr file folder (.npy files)
    You don't have to make dir of hr_path, mr_path, lr_path before execution
    """
    
    def create_folder(dir):
        r"""
        create folder(directory path + name to create)
        raises error when OSError occured.
        """
        try :
            if not os.path.exists(dir):
                os.makedirs(dir)
                
        except OSError:
            print('Error : Creating directory. ' + dir)
        
    create_folder(hr_path)
    create_folder(mr_path)
    create_folder(lr_path)
    
    orig_files = os.listdir(orig_path)
    orig_files = natsort.natsorted(orig_files)
    orig_files = [orig_path+'/'+name for name in orig_files]
    
    for step, path in enumerate(orig_files):
        file = Image.open(path)
        image_gray = transforms.functional.rgb_to_grayscale(file)
        hr = image_gray.resize((640,480))
        mr = image_gray.resize((320,240))
        lr = image_gray.resize((160,120))
        
        np_hr = np.asarray(hr)
        np_mr = np.asarray(mr)
        np_lr = np.asarray(lr)

        np.save(hr_path+'/'+str(step), np_hr)
        np.save(mr_path+'/'+str(step), np_mr)
        np.save(lr_path+'/'+str(step), np_lr)
        
        if (step-1) % 100 == 0:
            print(f"Step: {step} / {len(orig_files)}")

if __name__ == '__main__':
    
    orig_path = '/Users/healingmusic/Programming_local/testserver/orig'
    hr_path = '/Users/healingmusic/Programming_local/testserver/result/low'
    mr_path = '/Users/healingmusic/Programming_local/testserver/result/mid'
    lr_path = '/Users/healingmusic/Programming_local/testserver/result/high'
    
    jpg_to_numpy(orig_path, hr_path, mr_path, lr_path)
    
    # Implement like this 3 times for Train, Valid, Test