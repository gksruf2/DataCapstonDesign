import os
import natsort

# to merge train / test files
def step_files(dir, extension, step):
    r"""
    Step 'ALL' Files in directory
    ex) 0.npy, 1.npy -> 2.npy, 3.npy
    """
    files = os.listdir(dir)
    files = natsort.natsorted(files, reverse = True)
    
    if extension[0] != '.':
        extension = '.' + extension
        
    for file in files:
        temp = file # 10.npy
        temp = temp.replace(extension, '') # 10
        temp = int(temp) + step
        changed_name = str(temp) + extension
        os.rename(dir + '/' + file, dir + '/' + changed_name)

dir = '' # directory of test folder
step_files(dir, 'jpg', 10749) # start number : 10749 (right after last element of train folder)