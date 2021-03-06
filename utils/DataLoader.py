import os

import cv2
import numpy
import torch


class DataLoader():
    def __init__(self, path:str):
        self.root = path

    def load_image(self):
        '''
        以灰度图加载
        :return:
        '''
        x = []
        y = []
        file_list = os.listdir(os.path.join(self.root, "train_data"))
        train_path = os.path.join(self.root, "train_data")
        for filename in file_list:
            path = os.path.join(train_path,filename)
            image = cv2.imread(path,flags=cv2.IMREAD_GRAYSCALE)
            image = numpy.reshape(image,(image.shape[0],image.shape[1],1))  # 灰度确实一个通道，标准化
            x.append(image)
        x = numpy.array(x)

        file_list = os.listdir(os.path.join(self.root, "train_label"))
        train_path = os.path.join(self.root, "train_label")
        for filename in file_list:
            path = os.path.join(train_path,filename)
            image = cv2.imread(path,flags=cv2.IMREAD_GRAYSCALE)
            image = numpy.reshape(image, (image.shape[0], image.shape[1], 1))  # 灰度确实一个通道，标准化
            y.append(image)
        y = numpy.array(y)

        return x, y

    def load_preproces(self,path):
        return torch.load(os.path.join(path,"data.h5"))

    def save_preproces(self,path,data):
        torch.save(data,os.path.join(path,"data.h5"))


