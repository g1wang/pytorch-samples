import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
n1 = torch.linspace(-1, 1, 100)
print(x)
print(n1)
# x = torch.linspace(-1, 1, 100)
y = x.pow(2) + 0.2 * torch.rand(x.size())


# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)  # define the network
print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.2, momentum=0.8)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=0.005, alpha=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99))
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()

for t in range(500):
    prediction = net(x)

    loss = loss_func(prediction, y)  # input x and predict based on x
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'b-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
