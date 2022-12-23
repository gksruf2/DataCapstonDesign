import os
import shutil

def movefiles_by_name(dep, dest, names : list = []):
    r"""
    dep  : files to move from (departure)
    dest : files to move to (destination)
    nums : 
    file's name to move (with extender i.e : "123.jpg")
    default : move all files
    """
    
    def create_folder(dir):
        try :
            if not os.path.exists(dir):
                os.makedirs(dir)
            
        except OSError:
            print('Error : Creating directory. ' + dir)
    
    create_folder(dest)
    dep_files = os.listdir(dep)
    
    if len(names) != 0:
        for file in dep_files:
            if file in names:
                shutil.move(dep + '/' + file, dest + '/' + file)
    else:
        for file in dep_files:
            shutil.move(dep + '/' + file, dest + '/' + file)

train_list = [str(num) + '.jpg' for num in range(8893)]
val_list = [str(num) + '.jpg' for num in range(8893, 9893)]
test_list = [str(num) + '.jpg' for num in range(9893, 11893)]

# fill this out
# orig_path : directory contains .jpg files merged
# train/val/test path : paths to move files by train / validation / test order
orig_path = '' 
train_path = ''
val_path = ''
test_path = ''

movefiles_by_name(orig_path, train_path, train_list)
movefiles_by_name(orig_path, val_path, val_list)
movefiles_by_name(orig_path, test_path, test_list)