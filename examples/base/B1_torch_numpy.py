import torch
import numpy as np

# convert numpy to tensor or vise versa
np_data = np.arange(6).reshape((2, 3))
print("\nnp_data:", np_data)
torch_data = torch.from_numpy(np_data)
print("\ntorch_data:", torch_data)
tensor2array = torch_data.numpy()
print("\ntensor2array:", tensor2array)

# abs
data = [-1, -2, 1, 3]
tensor = torch.FloatTensor(data)
print('\nabs', '\nnumpy:', np.abs(data), '\ntorch:', torch.abs(tensor))

# sin
print('\nabs', '\nnumpy:', np.sin(data), '\ntorch:', torch.sin(tensor))

# mean
print('\nabs', '\nnumpy:', np.mean(data), '\ntorch:', torch.mean(tensor))

# matrix mutiplication
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
# correct method
print('\nmatrix multiplication (matmul)', '\nnumpy: ', np.matmul(data, data), '\ntorch: ', torch.mm(tensor, tensor))
# incorrect method
data = np.array(data)
print('\ndata:', data)
print('\nmatrix multiplication (dot)', '\nnumpy: ', data.dot(data))
# '\ntorch: ', tensor.dot(tensor))
data1 = [1, 2, 3, 4]
tensor1 = torch.FloatTensor(data1)
print(torch.dot(tensor1, tensor1))
print(tensor1.dot(tensor1))
