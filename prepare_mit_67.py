import os
import subprocess

def prepareMIT67():

    # download and unzip data
    os.system('wget http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar')
    os.system('wget http://web.mit.edu/torralba/www/TrainImages.txt')
    os.system('wget http://web.mit.edu/torralba/www/TestImages.txt')
    os.system('tar -xvf indoorCVPR_09.tar')

    os.mkdir('mit-67')
    all_dir_names = [name for name in os.listdir("Images") if os.path.isdir('Images')]

    # create train directory
    with open('TrainImages.txt') as file:
        traindata = file.readlines()
        traindata = [line.rstrip() for line in traindata]
    
    os.mkdir('mit-67/train')
    for dir_name in all_dir_names:
        os.mkdir(os.path.join('mit-67/train', dir_name))

    for train_sample in traindata:
        print('Processing', train_sample)
        source_path = os.path.join('Images', train_sample)
        des_path = os.path.join('mit-67/train', train_sample.split('/')[0])
        subprocess.run(['mv', source_path, des_path])

    # create test and val directory
    with open('TestImages.txt') as file:
        testdata = file.readlines()
        testdata = [line.rstrip() for line in testdata]

    os.mkdir('mit-67/val')
    for dir_name in all_dir_names:
        os.mkdir(os.path.join('mit-67/val', dir_name))

    for test_sample in testdata:
        print('Processing', test_sample)
        source_path = os.path.join('Images', test_sample)
        des_path = os.path.join('mit-67/val', test_sample.split('/')[0])
        subprocess.run(['mv', source_path, des_path])
    os.system('cp -r mit-67/val mit-67/test')

    os.system('mv mit-67 data')
    os.system('rm indoorCVPR_09.tar')
    os.system('rm -r Images')
    os.system('rm TrainImages.txt')
    os.system('rm TestImages.txt')

