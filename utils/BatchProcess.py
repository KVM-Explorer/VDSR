import utils.Optimizer as Optimizer
import utils.Net as Net


def train(data, model: Net.VDSR, epochs, optimizer: Optimizer.SGD, batch_size=16, ratio=0.7):
    x, y = data
    # 分配数据集
    train_x = x[:ratio * len(x), :, :, :]
    valid_x = x[ratio * len(x):, :, :, :]

    train_y = y[:ratio * len(x), :, :, :]
    valid_y = y[ratio * len(x):, :, :, :]

    # epoch
    for i in range(epochs):
        # batch
        for j in range(0, len(train_x), batch_size):
            train_x_batch = train_x[j:min(len(train_x), j + batch_size)]
            train_y_batch = train_y[j:min(len(train_y), j + batch_size)]
            ret = model.forward(train_x_batch)
            delta, loss = optimizer.solve(ret, train_y_batch)
            model.backward(delta)
            model = optimizer.update(model)

        loss = valid(data=(valid_x, valid_y), model=model, optimizer=optimizer)
        print(f"epoch:{i}/{epochs} loss:{loss}")

def valid(data, model: Net.VDSR, optimizer: Optimizer.SGD):
    x, y = data
    tot_num = len(x)
    total_loss = 0
    for i in range(len(x)):
        ret = model.forward(x[i, :, :, :])
        _, loss = optimizer.solve(ret, y[i, :, :, :])
        total_loss += loss

    total_loss /= tot_num
    return  total_loss
