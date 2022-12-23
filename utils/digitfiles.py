import os

def is_valid(dir : str):
    # dir_files = os.listdir(dir)
    # n_files = len(dir_files)
    pass


def change_all_files(*dirs : str):
    r"""
    input : name of directory
    function : change all file names by 1, 2, 3.jpg ascending order
    output : None
    """
    
    for dir in dirs:
        file_names = os.listdir(dir)

        i = 1
        for name in file_names:
            src = os.path.join(dir, name)
            dst = str(i) + '.jpg'
            dst = os.path.join(dir, dst)
            os.rename(src, dst)
            i += 1
            
def step_all_files(dir, extension, step):
    r"""
    change all file names by add step
    
    ex) 0.jpg, 1.jpg, 2.jpg -> 2.jpg, 3.jpg, 4.jpg (if step == 2)
    
    if there are not digit-named file, Error occured
    all files have to have same extension
    """
    # if extension[0] != '.':
    #     extension = '.' + extension
    
    # files = os.listdir(dir)
    # files_no_extension = [file.replace(extension, '') for file in files]
    # files_step = [str(int(file) + step) + extension) for file in files_no_extension]
    
    file_names = os.listdir(dir)

    i = 1
    for name in file_names:
        src = os.path.join(dir, name)
        dst = str(i) + '.jpg'
        dst = os.path.join(dir, dst)
        os.rename(src, dst)
        i += 1
            
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
        
def move_files(dep, dest, names : list = []):
    r"""
    dep  : files to move from
    dest : files to move to
    nums : 
    file's name to move (with extender i.e : "123.jpg")
    default : move all files
    """
    import shutil
    
    create_folder(dest)
    dep_files = os.listdir(dep)
    
    if len(names) != 0:
        for file in dep_files:
            if file in names:
                shutil.move(dep + '/' + file, dest + '/' + file)
    else:
        for file in dep_files:
            shutil.move(dep + '/' + file, dest + '/' + file)
            
def copy_files(dep, dest, names : list = []):
    r"""
    dep  : files to copy from
    dest : files to copy to
    nums : 
    file's name to move (with extender i.e : "123.jpg")
    default : copy all files
    """
    import shutil
    
    create_folder(dest)
    
    dep_files = os.listdir(dep)
    
    if len(names) != 0:
        for file in dep_files:
            if file in names:
                shutil.copy2(dep + '/' + file, dest + '/' + file)
    else:
        for file in dep_files:
            shutil.copy2(dep + '/' + file, dest + '/' + file)
        
def make_file_names_with_range(start : int, end : int, extender : str):
    r"""
    input : make_file_names_with_range(3, 6, 'jpg')
    output : ['3.jpg', '4.jpg', '5.jpg'] # not include 6
    """
    if not extender.startswith('.'):
        extender = '.' + extender
    
    files = []
    for i in range(end - start):
        files.append(str(start + i) + extender)
    return files

def merge_dirs_by_move(dir1 : str, dir2 : str, dest : str):
    r"""
    merge dir1 and 2 into dest directory
    after merge, dir 1 and 2 are removed
    * warning) cannot gurantee if there are other directories in dir 1 & 2
    """
    import shutil
    
    create_folder(dest)
    
    dir1_files = os.listdir(dir1)
    dir2_files = os.listdir(dir2)
    
    for file in dir1_files:
        shutil.move(dir1 + '/' + file, dest + '/' + file)
    
    for file in dir2_files:
        shutil.move(dir2 + '/' + file, dest + '/' + file)
        
    os.rmdir(dir1)
    os.rmdir(dir2)
    
def merge_dirs_by_copy(dir1 : str, dir2 : str, dest : str):
    r"""
    copy dir1 and 2 into dest directory
    after merge, dir 1 and 2 are 'not' removed
    * warning) cannot gurantee if there are other directories in dir 1 & 2
    """
    import shutil
    
    create_folder(dest)
    
    dir1_files = os.listdir(dir1)
    dir2_files = os.listdir(dir2)
    
    for file in dir1_files:
        shutil.copy2(dir1 + '/' + file, dest + '/' + file)
    
    for file in dir2_files:
        shutil.copy2(dir2 + '/' + file, dest + '/' + file)
        
def step_files(dir, step):
    pass
        

    
    

if __name__ == '__main__':
    
    # make_file_names_with_range testcode
    result = make_file_names_with_range(10, 15, 'jpg')
    print(result)
    
    result = make_file_names_with_range(10, 15, '.jpg')
    print(result)

    
# from google.colab import drive
# drive.mount('/content/drive')