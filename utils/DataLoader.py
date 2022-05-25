import os

import cv2
import numpy
import torch


class DataLoader():
    def __init__(self, path:str):
        self.root = path

    def load_image(self):
        x = []
        y = []
        file_list = os.listdir(os.path.join(self.root, "train_data"))
        train_path = os.path.join(self.root, "train_data")
        for filename in file_list:
            path = os.path.join(train_path,filename)
            image = cv2.imread(path)
            x.append(image)
        x = numpy.array(x)

        file_list = os.listdir(os.path.join(self.root, "train_label"))
        train_path = os.path.join(self.root, "train_label")
        for filename in file_list:
            path = os.path.join(train_path,filename)
            image = cv2.imread(path)
            y.append(image)
        y = numpy.array(y)

        return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32)

    def load_preproces(self,path):
        return torch.load(os.path.join(path,"data.h5"))

    def save_preproces(self,path,data):
        torch.save(data,os.path.join(path,"data.h5"))


