import csv
import cv2
import os
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from os.path import join


def read_image_tensor(f_path, permute=True):
    img = cv2.imread(f_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    if permute:
        img = np.transpose(img, (2,0,1))
    return torch.from_numpy(img).unsqueeze(0)

class AttackDataset(Dataset):
    def __init__(self):
        self.root_dir = './datas/images'
        
        self.annos = []
        with open('./datas/dev_dataset.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                self.annos.append({'ImageId': row['ImageId'], 'TrueLabel': row['TrueLabel']})

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):

        anno = self.annos[idx]
        f_path = join(self.root_dir, anno['ImageId']+'.png')
        img = cv2.imread(f_path) #, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))

        label = int(anno['TrueLabel']) -1 

        return img, label, anno['ImageId']

