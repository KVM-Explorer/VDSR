import os

import cv2
import torch


class DataLoader():
    def __init__(self, train_path:str, label_path:str):
        self.train_path = train_path
        self.label_path = label_path

    def load(self):
        x = []
        y = []
        file_list = os.listdir(self.train_path)
        for filename in file_list:
            path = os.path.join(self.train_path,filename)
            image = cv2.imread(path)
            x.append(image)

        file_list = os.listdir(self.label_path)
        for filename in file_list:
            path = os.path.join(self.train_path,filename)
            image = cv2.imread(path)
            y.append(image)

        return torch.tensor(x), torch.tensor(y)
