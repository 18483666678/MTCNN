import torch

a = torch.Tensor([[1,2],[3,4],[5,6]])

b = torch.Tensor([0,0,1])  #类别  可以当成a的索引

center = torch.Tensor([[1,1],[2,2]]) #两个中心

#[[1,1],[1,1],[2,2]]
center_exp = center.index_select(dim=0,index=b.long())#b变成longfloat形状
print(center_exp)

count = torch.histc(b,bins=2,min=0,max=2)
print(count)

num = count.index_select(dim=0,index=b.long())
print(num)

#求期望
x = (a-center_exp)**2
print(x)
x = torch.sum((a-center_exp)**2,dim=1)
print(x)
x = torch.sqrt(torch.sum((a-center_exp)**2,dim=1))
print(x)
x = torch.sqrt(torch.sum((a-center_exp)**2,dim=1))/num
print(x)
loss = torch.sum(torch.sqrt(torch.sum((a-center_exp)**2,dim=1))/num) / 2
print(loss)

# print("-------------------------------------")
# print(a.size())
# print(b.size())
# print(center.size())
# print(center_exp.size())
# print(count.size())
# print(num.size())