import torch
import numpy as np


# a = torch.normal(mean=0.5, std=torch.arange(1., 6.))
# print(a)
#
# a = np.array([1,2])
# b = a[None]
# print(a)
# print(b)

a = np.array([[1,2],[3,4],[5,6],[1,2],[3,4],[5,6]])   #(3,2)
print(a.shape)
b = np.array([0,1,2,1,2,1]) #(3,)
for i in range(3):
    x,y = a[b==i,0],a[b==i,1]
    print(i,x,y)


import torch
from torch.autograd import Variable
import numpy as np

data = Variable(torch.from_numpy(np.array([3, 2, 1, 8, 7], dtype=np.float32)), requires_grad=True)
idx = Variable(torch.from_numpy(np.array([0, 0, 1, 1, 2], dtype=np.int64)), requires_grad=False)

# out = Variable(torch.zeros(3), requires_grad=True)
# print(out)
# out.scatter_add_(0, idx, data)
# print(out)
out = Variable(torch.zeros(3), requires_grad=True).clone()
print(out)
out = out.scatter_add_(0, idx, data)
print(out)
out = out.scatter_add_(0, idx, data)
print(out)