import os

import torch
import utils.Net as Net
import utils.DataLoader as DataLoader
import utils.PreProcess as PreProcess
import utils.BatchProcess as BatchProcess
import utils.Optimizer as Optimizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":

    # 数据加载
    dataloader = DataLoader.DataLoader("./data")

    if not os.path.getsize("./data/preprocess"):
        x,y = dataloader.load_preproces("data/preprocess")
    else:
        x, y = dataloader.load_image()

        # 预处理
        x = PreProcess.normalize(x, (64, 64))
        y = PreProcess.normalize(y, (64, 64))

        # 暂存数据集到本地
        dataloader.save_preproces("data/preprocess", data=(x, y))


    if torch.cuda.is_available():
        x = x.to(device='cuda')
        y = y.to(device='cuda')

    # 初始化模型
    net = Net.VDSR()

    # 数据训练连
    BatchProcess.train(data=(x, y),
                       model=net,
                       epochs=10,
                       batch_size=4,
                       optimizer=Optimizer.SGD(None))

    # generate and show SR Image

