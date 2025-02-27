import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir


# borrow the code from https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py
def prepareTinyImageNet():

    # download and unzip data
    os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
    os.system('unzip tiny-imagenet-200.zip')
    os.system('mv tiny-imagenet-200 data')
    os.system('rm tiny-imagenet-200.zip')

    target_folder = './data/tiny-imagenet-200/val/'
    test_folder   = './data/tiny-imagenet-200/test/'
    
    # os.mkdir(test_folder)
    val_dict = {}
    with open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]
    
    paths = glob.glob('./data/tiny-imagenet-200/val/images/*')
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')
        if not os.path.exists(test_folder + str(folder)):
            os.mkdir(test_folder + str(folder))
            os.mkdir(test_folder + str(folder) + '/images')

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        
        if len(glob.glob(target_folder + str(folder) + '/images/*')) < 25:
            dest = target_folder + str(folder) + '/images/' + str(file)
        else:
            dest = test_folder + str(folder) + '/images/' + str(file)
        move(path, dest)
    
    rmdir('./data/tiny-imagenet-200/val/images')
    os.system('rm -r ./data/tiny-imagenet-200/test/images')
