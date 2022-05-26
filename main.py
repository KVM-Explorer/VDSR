import cv2
import torch
import utils.Net as Net
import utils.DataLoader as DataLoader
import utils.PreProcess as PreProcess
import utils.BatchProcess as BatchProcess
import utils.Optimizer as Optimizer
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == "__main__":

    # 数据加载
    dataloader = DataLoader.DataLoader("./data")

    if not os.path.getsize("./data/preprocess"):
        x, y = dataloader.load_preproces("data/preprocess")
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
    model = BatchProcess.train(data=(x, y),
                               model=net,
                               epochs=3,
                               batch_size=8,
                               optimizer=Optimizer.SGD(None))

    # generate and show SR Image
    localtime = time.strftime("%Y-%m-%dT%H_%M_%S", time.localtime(time.time()))
    model.save(f"model/{localtime}.h5")

    # 加载模型
    # model = Net.load_model("")

    # 图片输入输出测试
    image = cv2.imread("data/test_data/1.png")
    norm = PreProcess.test_normalize(image,(64,64))
    ret = model.forward(norm)
    show_image = PreProcess.reverse_data(ret)
    cv2.imshow("result",show_image)
    cv2.imwrite("result.png",show_image)
    cv2.waitKey(0)
