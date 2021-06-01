from torch.utils.data import Dataset, DataLoader
from os.path import join
import numpy as np
from itertools import accumulate
from torch import Tensor as T
import torch
from tqdm import tqdm
import h5py
from random import randrange

class DualLPCData(Dataset):
    def __init__(self, h5_path='./data/p225.h5', length=256):
        self.dataset = h5py.File(h5_path, 'r')
        self.names = list(self.dataset.keys())
        self.length = length
        self.__num__ = len(self.names)
    def __getitem__(self, i):
        ni = self.names[i]
        di = self.dataset[ni]
        ti = randrange(di.shape[0] - self.length + 1)
        X = di[ti : ti + self.length, 0, :]
        Y = di[ti : ti + self.length, 1, :]
        return X, Y
    def __len__(self):
        return self.__num__

if __name__ == '__main__':
    dataset = DualLPCData()
    #for i in tqdm(range(dataset.__len__())):
    #    x, y = dataset.__getitem__(i)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    for x, y in tqdm(dataloader):
        continue
