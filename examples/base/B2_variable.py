import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
# till now the tensor and variable seem the same.
# However, the variable is a part of the graph, it's a part of the auto-gradient.
variable = Variable(tensor, requires_grad=True)
print(tensor)
print(variable)
t_out = torch.mean(tensor*tensor)
print(t_out)
v_out = torch.mean(variable*variable)
print(v_out)
v_out.backward()
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
print(variable.grad)
print(variable)