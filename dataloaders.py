import numpy as np
import pandas as pd
import os
import requests
import random
import pickle
from PIL import Image
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torchvision import datasets
from prepare_tiny_imagenet import prepareTinyImageNet
from prepare_mit_67 import prepareMIT67
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset


# borrow the code from https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
class Cub2002011Dataset(Dataset):
    base_folder = 'CUB_200_2011/images'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    
    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
    
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def get_confirm_token(self, response):

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        
        return None

    def save_response_content(self, response, destination):
        
        CHUNK_SIZE = 32768
        
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    def download_file_from_google_drive(self, in_id, des):
        
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        
        response = session.get(URL, params = { 'id' : in_id }, stream = True)
        token = self.get_confirm_token(response)
        
        if token:
            params = { 'id' : in_id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        
        self.save_response_content(response, des)

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # download_file_from_google_drive(self.file_id, self.root, self.filename)
        self.download_file_from_google_drive(self.file_id, os.path.join(self.root, self.filename))
        print('Files are downloaded!')
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, data_root, split, transform=None, download=True):
        super(CIFAR10Dataset, self).__init__(root=data_root, train=split, transform=transform, download=download)
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR100Dataset(datasets.CIFAR100):
    def __init__(self,  data_root, split, transform=None, download=True):
        super(CIFAR100Dataset, self).__init__(root=data_root, train=split, transform=transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_mit67_dataset(dataset_name, mode, num_class, labelTable):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # data augmentation
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # download and prepare MIT-67
    if not os.path.exists('data/mit-67/'):
        prepareMIT67()

    dataset = None
    if dataset_name.startswith('subclass'):
        dir_path = os.path.join('data/mit-67', mode)
        dataset = datasets.ImageFolder(dir_path, transform=transform)

        '''
        dataset.targets = np.array(dataset.targets)
        idx_mask = (dataset.targets == 3) | (dataset.targets == 20) | (dataset.targets == 40) | (dataset.targets == 15) | (dataset.targets == 30)

        # change labels to consecutive labels
        tmp = []
        for idx in range(idx_mask.shape[0]):
            if idx_mask[idx]:
                tmp.append((dataset.samples[idx][0], labelTable[dataset.samples[idx][1]]))
        dataset.samples = tmp
        '''

        # change labels to consecutive labels
        num_data = np.array(dataset.targets).shape[0]
        
        tmp = []
        for idx in range(num_data):
            if dataset.samples[idx][1] in list(labelTable.keys()):
                tmp.append((dataset.samples[idx][0], labelTable[dataset.samples[idx][1]]))
        dataset.samples = tmp

        dataset.num_classes = num_class
    else:
        dir_path = os.path.join('data/mit-67', mode)
        dataset = datasets.ImageFolder(dir_path, transform=transform)
        dataset.num_classes = num_class

    return dataset


def get_cub200_dataset(dataset_name, mode, num_class, labelTable):

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # data augmentation
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_flag = True
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_flag = False

    dataset = None
    if dataset_name.startswith('subclass'):

        dataset = Cub2002011Dataset('./data/', train=train_flag, transform=transform, download=True)

        # drop rows
        for id_num in range(200):
            if id_num not in labelTable:
                dataset.data.drop(dataset.data[dataset.data['target'] == id_num + 1].index, inplace=True)

        # change labels to consecutive labels
        #for key in [70, 8, 2, 26, 107]:
        for key in list(labelTable.keys()):
            dataset.data.loc[(dataset.data.target == key + 1), 'target'] = labelTable[key] + 1

        dataset.num_classes = num_class
    else:
        dataset = Cub2002011Dataset('./data/', train=train_flag, transform=transform, download=True)
        dataset.num_classes = num_class

    return dataset


def get_tinyimagenet_dataset(dataset_name, mode, num_class, labelTable):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # data augmentation
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])

    # download and prepare Tiny ImageNet
    if not os.path.exists('data/tiny-imagenet-200/'):
        prepareTinyImageNet()

    dataset = None
    if dataset_name.startswith('subclass'):
        dir_path = os.path.join('data/tiny-imagenet-200', mode)
        dataset = datasets.ImageFolder(dir_path, transform=transform)

        '''
        dataset.targets = np.array(dataset.targets)
        idx_mask = (dataset.targets == 99) | (dataset.targets == 131) | (dataset.targets == 168) | (dataset.targets == 138) | (dataset.targets == 139)

        # change labels to consecutive labels
        tmp = []
        for idx in range(idx_mask.shape[0]):
            if idx_mask[idx]:
                tmp.append((dataset.samples[idx][0], labelTable[dataset.samples[idx][1]]))
        dataset.samples = tmp
        '''
        # change labels to consecutive labels
        num_data = np.array(dataset.targets).shape[0]
        tmp = []
        for idx in range(num_data):
            if dataset.samples[idx][1] in list(labelTable.keys()):
                tmp.append((dataset.samples[idx][0], labelTable[dataset.samples[idx][1]]))
        dataset.samples = tmp

        dataset.num_classes = num_class
    else:
        dir_path = os.path.join('data/tiny-imagenet-200', mode)
        dataset = datasets.ImageFolder(dir_path, transform=transform)
        dataset.num_classes = num_class
    
    return dataset


def get_cifar100_dataset(dataset_name, mode, num_class, labelTable):

    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    
    # Data augmentation
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        split = True
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        split = False

    dataset = None
    if dataset_name.startswith('subclass'):
        dataset = CIFAR100Dataset('data/' + dataset_name.split('-')[-1], split=split, transform=transform, download=True)
        dataset.targets = torch.tensor(dataset.targets)

        #idx_mask = (dataset.targets == 55) | (dataset.targets == 35) | (dataset.targets == 72) | (dataset.targets == 10) | (dataset.targets == 11)
        idx_mask = torch.zeros(len(dataset), dtype=torch.bool)
        for key in list(labelTable.keys()):
            indices = (dataset.targets == key).nonzero(as_tuple=True)[0]
            idx_mask[indices] = True
        dataset.targets = dataset.targets[idx_mask]

        # change labels to consecutive labels
        for key in list(labelTable.keys()):
            dataset.targets[dataset.targets == key] = labelTable[key]

        dataset.data = dataset.data[idx_mask.numpy().astype(bool)]
        dataset.num_classes = num_class
    else:
        dataset = CIFAR100Dataset('data/' + dataset_name, split=split, transform=transform, download=True)
        dataset.num_classes = num_class

    return dataset


def get_cifar10_dataset(dataset_name, mode, num_class, labelTable):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # Data augmentation
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        split = True
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        split = False

    dataset = None
    if dataset_name.startswith('subclass'):
        dataset = CIFAR10Dataset('data/' + dataset_name.split('-')[-1], split=split, transform=transform, download=True)
        dataset.targets = torch.tensor(dataset.targets)

        # idx_mask = (dataset.targets == 3) | (dataset.targets == 5) | (dataset.targets == 2) | (dataset.targets == 4) | (dataset.targets == 0)
        idx_mask = torch.zeros(len(dataset), dtype=torch.bool)
        for key in list(labelTable.keys()):
            indices = (dataset.targets == key).nonzero(as_tuple=True)[0]
            idx_mask[indices] = True
        dataset.targets = dataset.targets[idx_mask]
        
        # change labels to consecutive labels
        for key in list(labelTable.keys()):
            dataset.targets[dataset.targets == key] = labelTable[key]
        dataset.data = dataset.data[idx_mask.numpy().astype(bool)]

        dataset.num_classes = num_class
    else:
        dataset = CIFAR10Dataset('data/' + dataset_name, split=split, transform=transform, download=True)
        dataset.num_classes = num_class

    return dataset


def get_dataloader(dataset, batch_size=1, shuffle=True, mode='train', num_workers=4, class_num=10, labelTable=None):
    
    if dataset.split('-')[-1] == 'cifar10':
        dataset = get_cifar10_dataset(dataset, mode, class_num, labelTable)
    elif dataset.split('-')[-1] == 'cifar100':
        dataset = get_cifar100_dataset(dataset, mode, class_num, labelTable)
    elif dataset.split('-')[-1] == 'tinyimagenet':
        dataset = get_tinyimagenet_dataset(dataset, mode, class_num, labelTable)
    elif dataset.split('-')[-1] == 'cub2002011':
        dataset = get_cub200_dataset(dataset, mode, class_num, labelTable)
    elif dataset.split('-')[-1] == 'mit67':
        dataset = get_mit67_dataset(dataset, mode, class_num, labelTable)
    else:
        raise Exception(f'Does not support {dataset}!')

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


# test dataloader
#train_loader = get_dataloader(dataset='fruits360', mode='train', batch_size=1, num_workers=2)
#trueLabel2fakeLabel = {3: 0, 1: 1, 8: 2}
#train_loader = get_dataloader(dataset='cifar10', mode='train', batch_size=1, num_workers=2)
#train_loader = get_dataloader(dataset='subclass-cifar100', mode='eval', batch_size=2048, num_workers=0, shuffle=False, class_num=3, labelTable=trueLabel2fakeLabel)

#d1, t1 = next(iter(train_loader))
#print(t1)

#t1 = get_dataloader(dataset='tinyimagenet', mode='train', batch_size=1, num_workers=0, class_num=200)
#t2 = get_dataloader(dataset='subclass-tinyimagenet', mode='train', batch_size=1, num_workers=0, class_num=3, labelTable={120: 0, 153: 1, 199: 2})

#d1 = next(iter(t1))
#data2 = next(iter(t2))

#print(d1)
#print(data2)
#print(data1[0][0,:,:,:].to('cuda:0') )
#print(data2[0][0,:,:,:].to('cuda:0') )

#train_loader = get_dataloader('cub2002011', mode='train', batch_size=1, num_workers=0, class_num=200)
#train_loader = get_dataloader('subclass-cub2002011', mode='train', batch_size=1, num_workers=0, class_num=3, labelTable={30: 0, 190: 1, 88: 2})
'''
print(len(train_loader))
for _ in range(len(train_loader)):
    data = next(iter(train_loader))
    print(data[0].size())
    print(data[1])
'''
