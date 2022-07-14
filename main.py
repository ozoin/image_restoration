from dataset import ImageRestorationDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
import config
import numpy as np


def CreateLoaders():
    dset = ImageRestorationDataset(img_dir='data')

    val_len = int(len(dset) * config.VAL_RATIO)
    val_idcs = np.arange(len(dset) - val_len, len(dset))
    train_idcs = np.arange(0, len(dset) - val_len)

    train_set = Subset(dset, train_idcs)
    val_set = Subset(dset, val_idcs)

    full_dloader = DataLoader(dset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=4,
                              shuffle=False)

    train_dloader = DataLoader(train_set,
                               batch_size=config.BATCH_SIZE,
                               num_workers=1,
                               shuffle=False)

    val_dloader = DataLoader(val_set,
                             batch_size=config.BATCH_SIZE,
                             num_workers=1,
                             shuffle=False)

    return full_dloader, train_dloader, val_dloader
