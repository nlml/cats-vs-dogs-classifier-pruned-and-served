from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL.Image import BILINEAR
import os
import numpy as np
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import warnings

# Ignore annoying PIL warnings 
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def split_dataset(dataset_dir, rng, p_train, p_val, p_test):
    '''Split a folder containing a dataset into random train, validation and test portions.
    Args:
        dataset_dir (str): path containing a folder of images (or other inputs) for each class.
        rng (np.random.RandomState): numpy random number generator object.
        p_train (float): proportion of data to training set.
        p_val (float): ... to validation set.
        p_test (float): ... to test set.
    Returns:
        imagepaths (dict(set)): a dict with keys=['train', 'test', 'val'], each key contains a set
            that contains the strings <className>/filename.ext belonging to that subset.
        classes (list(str)): the classnames, derived from the folders and ordered alphabetically.
    '''
    classes = list(sorted(os.listdir(dataset_dir)))
    imagefilenames = {}
    for cls in classes:
        imagefilenames[cls] = np.array(os.listdir(dataset_dir / cls))
    imagepaths = {}
    for cls in imagefilenames.keys():
        perm = rng.permutation(len(imagefilenames[cls]))
        n = {}
        n['train'] = int(len(perm) * p_train)
        n['val'] = int(len(perm) * p_val)
        n['test'] = len(perm) - n['val'] - n['train']
        for train_test_val in n.keys():
            if train_test_val not in imagepaths:
                imagepaths[train_test_val] = []
            this = [cls + '/' + i for i in imagefilenames[cls][perm[:n[train_test_val]]]
                    if i.endswith('.jpg')]
            imagepaths[train_test_val] += this
    for k in imagepaths:
        imagepaths[k] = set(imagepaths[k])
    return imagepaths, classes


def get_transforms():
    imagenet_norm_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomAffine(
            degrees=(-20, 20),
            translate=(0.15, 0.15),
            resample=BILINEAR
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomGrayscale(0.03),
        transforms.ToTensor(),
        imagenet_norm_transform,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        imagenet_norm_transform,
    ])
    transform = {
        'train': train_transform,
        'test': test_transform,
        'val': test_transform
    }
    return transform


def get_torch_datasets(transform, imagepaths, data_dir, subdir='PetImages/'):
    dataset = {}
    for train_val_test in imagepaths.keys():
        dataset[train_val_test] = ImageFolder(
            str(data_dir / subdir),
            transform=transform[train_val_test],
            is_valid_file=lambda x: x.split(subdir)[-1] in imagepaths[train_val_test]
        )
    assert dataset['train'].classes == dataset['val'].classes == dataset['test'].classes
    return dataset


def get_torch_loaders(dataset, batch_size, num_workers=cpu_count()):
    loader = {}
    for train_val_test in dataset:
        loader[train_val_test] = DataLoader(
            dataset[train_val_test],
            num_workers=num_workers,
            batch_size=batch_size[train_val_test],
            shuffle=train_val_test == 'train'
        )
    return loader
