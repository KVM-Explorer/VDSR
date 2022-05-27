import torch

import utils.Optimizer as Optimizer
import utils.Net as Net
import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s -%(levelname)s - %(message)s')


def train(data, model: Net.VDSR, epochs, optimizer: Optimizer.SGD, batch_size=16, ratio=0.7):
    x, y = data
    # 分配数据集
    data_gap = int(ratio * len(x))
    train_x = x[:data_gap, :, :, :]
    valid_x = x[data_gap:, :, :, :]

    train_y = y[:data_gap, :, :, :]
    valid_y = y[data_gap:, :, :, :]

    # epoch
    for epoch in range(epochs):
        # batch
        for j in range(0, len(train_x), batch_size):
            train_x_batch = train_x[j:min(len(train_x), j + batch_size)]
            train_y_batch = train_y[j:min(len(train_y), j + batch_size)]
            ret = model.forward(train_x_batch)
            delta, loss = optimizer.solve(ret, train_y_batch)
            logging.info(f"Loss:{loss}")
            model.backward(delta)
            model = optimizer.update(model)

        loss = valid(data=(valid_x, valid_y), model=model, optimizer=optimizer)
        optimizer.update_learning_rate(epoch)
        logging.info(f"epoch:{epoch}/{epochs} loss:{loss} learning_rate:{optimizer.learning_rate}")
    return model


def valid(data, model: Net.VDSR, optimizer: Optimizer.SGD):
    logging.info("=================Start valid==================")
    x, y = data
    tot_num = len(x)
    total_loss = 0
    batches, channels, rows, cols = x.size()
    for i in range(len(x)):
        tmp_x = torch.reshape(x[i, :, :, :], (1, channels, rows, cols))
        ret = model.forward(tmp_x)
        _, loss = optimizer.solve(ret, y[i, :, :, :])
        total_loss += loss

    total_loss /= tot_num
    return total_loss
