import os
import torchvision.models as models
import torchvision.transforms as transforms
import torch

# 原始目录
ori_dataset_dir = 'D:\\all-dataset\\dogs-vs-cats\\train'
# 小数据及目录
small_data_dir = 'D:\\all-dataset\\dogs-vs-cats-small'
# 划分数据集:train|validation|test
train_dir = os.path.join(small_data_dir, 'train')
validation_dir = os.path.join(small_data_dir, 'validation')
test_dir = os.path.join(small_data_dir, 'test')
# cats dogs train
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
# cats dogs validation
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# cats dogs test
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

import os
import torch
import torch.utils.data as data
from PIL import Image


def default_loader(path, label):
    return Image.open(path).convert('RGB'), label


class MyDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        f = open(root, 'r')
        imgs = []
        for line in f.readlines():
            cols = line.strip('\n').rstrip().split()
            path = cols[0]
            label = cols[1]
            imgs.append((path, int(label)))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img, label = self.loader(path, label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # 数据加载
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.Resize(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset_train = MyDataset(root="../catvsdog-train.txt", transform=transform_train)
    dataLoader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=8, shuffle=True, num_workers=2)

    # 数据加载
    transform_val = transforms.Compose([ transforms.Resize(224),
                                        transforms.CenterCrop(224),transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset_val = MyDataset(root="../catvsdog-validation.txt", transform=transform_val)
    dataLoader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    denseNet = models.densenet161(pretrained=True)
    # for param in denseNet.parameters():
    #     param.requires_grad = False
    denseNet.classifier = torch.nn.Linear(2208, 2)

    denseNet = torch.nn.DataParallel(denseNet)
    # 损失函数
    loss_fun = torch.nn.CrossEntropyLoss()
    # 优化函数
    optimizer = torch.optim.Adam(denseNet.parameters())
    print(denseNet)
    denseNet.to(device)

    epochs = 100
    for epoch in range(epochs):
        loss_sum = 0
        acc_train_sum = 0
        for step_train, data in enumerate(dataLoader_train):
            batch_size = len(data)
            b_x, b_y = data
            b_x, b_y = b_x.to(device), b_y.to(device)
            outs = denseNet(b_x)
            _, predicted = torch.max(outs.data, 1)
            acc_train = float(predicted.eq(b_y.data).cpu().sum()) / float(b_y.size(0))
            acc_train_sum += acc_train
            loss = loss_fun(outs, b_y)
            loss_sum += loss.cpu().data.numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step_train + 1) % 10 == 0:
                print('epoch: %d ,step: %d, loss: %.4f, accuracy: %.4f' % (
                    (epoch + 1), (step_train + 1), loss.cpu().data.numpy(), acc_train))

        acc_val_sum = 0
        for step_val, data in enumerate(dataLoader_val):
            batch_size = len(data)
            b_x, b_y = data
            b_x, b_y = b_x.to(device), b_y.to(device)
            outs = denseNet(b_x)
            _, predicted = torch.max(outs.data, 1)
            acc_val = float(predicted.eq(b_y.data).cpu().sum()) / float(b_y.size(0))
            acc_val_sum += acc_val

        print('epoch: %d ,trian_loss:  %.4f ,train_acc: %.4f, val_acc: %.4f' % (
            (epoch + 1), (loss_sum / float(step_train + 1)), (acc_train_sum / float(step_train + 1)),
            (acc_val_sum / float(step_val + 1))))
    torch.save(denseNet, 'catvsdogs.pkl')  # save entire


# conv_base =
