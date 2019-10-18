import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class Trainset(Dataset):
    def __init__(self, dataset, label):
        self.trainset = dataset
        self.label = label - 30

    def __getitem__(self, item):
        data = self.trainset[item]
        target = self.label[item]
        return data, target

    def __len__(self):
        return len(self.label)


class Validset(Dataset):
    def __init__(self, dataset, valid_begin_time, valid_signed_time):
        self.trainset = dataset
        self.valid_begin_time = valid_begin_time
        self.valid_signed_time = valid_signed_time

    def __getitem__(self, item):
        data = self.trainset[item]
        valid_begin_time = self.valid_begin_time[item]
        valid_signed_time = self.valid_signed_time[item]
        return data, valid_begin_time, valid_signed_time

    def __len__(self):
        return len(self.trainset)


class Testset(Dataset):
    def __init__(self, dataset, test_begin_time):
        self.dataset = dataset
        self.test_begin_time = test_begin_time

    def __getitem__(self, item):
        data = self.dataset[item]
        test_begin_time = self.test_begin_time[item]
        return data, test_begin_time

    def __len__(self):
        return len(self.dataset)
