import torch
import torchvision
import os
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
import torch.utils.data as Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DOWNLOAD_MNIST = False
if not (os.path.exists('.D:/all-dataset/mnist/')) or not os.listdir('D:/all-dataset/mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
data_train = datasets.MNIST(root='D:/all-dataset/mnist/',
                            transform=transform,
                            train=True,
                            download=DOWNLOAD_MNIST)

data_test = datasets.MNIST(root='D:/all-dataset/mnist/',
                           transform=transform,
                           train=False)

data_loader_train = Data.DataLoader(dataset=data_train, batch_size=64, shuffle=True)

# data_loader_test = Data.DataLoader(dataset=data_test, batch_size=64, shuffle=True)
test_x = torch.unsqueeze(data_test.test_data, dim=1).type(torch.FloatTensor)[
         :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = data_test.test_labels[:2000]

images, labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1, 2, 0)
print([labels[i] for i in range(64)])


# plt.imshow(img)


# plt.show()


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 32, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


# torch.cuda.set_device(0)
model = Model()
model = torch.nn.DataParallel(model)
# 损失函数
loss_fun = torch.nn.CrossEntropyLoss()
# 优化函数
optimizer = torch.optim.Adam(model.parameters())

epochs = 10
for epoch in range(epochs):
    for step, (b_x, b_y) in enumerate(data_loader_train):
        # b_x, b_y = b_x.cuda(), b_y.cuda()
        b_x, b_y = b_x.to(device), b_y.to(device)
        outputs = model(b_x)
        loss = loss_fun(outputs, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = model(test_x.cuda())
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.5f' % accuracy)
torch.save(model, 'hand-write.pkl')  # save entire
