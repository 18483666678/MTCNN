import torch

#求的相似度是input1中的[1,2]与input2中的[3,4],input1中的[3,4]与input2中的[5,6].
input1 = torch.Tensor([[1,2],[2,3]])
input2 = torch.Tensor([[3,4],[5,6]])
cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
output = cos(input1,input2)
print(1)
d = cos(input2,input1)
print(d)
print(output)
pdist = torch.nn.PairwiseDistance(p=2)
c = pdist(input2,input1)
print(c)

#假如我们想求的是一次求出1中的[1,2]与2[[3,4],[5,6]]的相似性，
# 我们令a=input1,h=input2,先对a,h 求得二范数c,f，
# 然后复制c,f（广播法则）和a,h一样得d,g。d,g然后和a,h每个元素相除，
# 利用np.dot()就可得到余弦相似性。，
import numpy as np

print(2)
a = torch.Tensor([[1,2],[2,3]])
h = torch.Tensor([[3,4],[5,6]])
print(a)
print(h)

#求二范数  y = torch.norm(x, p)  -- p-范式  范数公式：||X||p = （|x1|**p + |x2|**p +···+|xn|**p）** 1/p
f = torch.norm(h,2,1,True)
print(torch.norm(h,2,1)) #不加True是一维 加Ture是二维  1是两个结果
print(f)

c = torch.norm(a,2,1,True)
g = f.expand_as(h)  #使求完范数的和之前的一样
d = c.expand_as(a)
print(c)
print(d)
print(g)
print(11111)

#div（）两张量input和other逐元素相除，并将结果返回到输出。即， \( out_i= input_i / other_i \)
#两张量形状不须匹配，但元素数须一致。 注意：当形状不匹配时，input的形状作为输出张量的形状。
l = h.div(g) #h和g的每个元素求除法
e = a.div(d)
print(e)
print(l)
print(22222)
# q = np.dot(e,l)
# #矩阵点乘dot（）：所得到的数组中的每个元素为，第一个矩阵中与该元素行号相同的元素
# # 与第二个矩阵与该元素列号相同的元素， 两两相乘后再求和。
# print(q)

j = torch.t(e) #转置  将矩阵的行列互换得到的新矩阵称为转置矩阵，转置矩阵的行列式不变
print(3)
print(j)
m = np.dot(l,e[0])  #e shi a,l shi h
print(m)
print(e[0])
print(2)
n = np.dot(l.numpy(),e[1].numpy())
print(n)
print(e[1])