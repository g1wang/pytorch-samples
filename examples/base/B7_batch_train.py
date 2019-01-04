import torch
import torch.utils.data as Data

torch.manual_seed(1)  # reproducible

BATCH_SIZE = 5

x = torch.linspace(1, 10, 100)
y = torch.linspace(10, 1, 100)

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # random shuffle for training 是否随机打乱顺序
    num_workers=2  # subprocesses for loading data 多线程读取数据的线程数
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data
            print('epoch: ', epoch, '|step:', step, '|batch x:', batch_x.numpy(), '|batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()
