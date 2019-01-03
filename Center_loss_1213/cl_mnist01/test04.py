import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import matplotlib.patheffects as PathEffects
import torch.optim.lr_scheduler as lr_scheduler

EPOCH = 1
BATCH_SIZE = 128

train_data = torchvision.datasets.MNIST("mnist_data/",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST("mnist_data/",train=False,transform=torchvision.transforms.ToTensor())

# train_loader = data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
#
# test_x = torch.autograd.Variable(torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255.)
# test_y = test_data.test_labels[:2000]
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,5,1,2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(32*7*7,2)
        self.fc2 = nn.Linear(2,10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0),-1)
        feat = self.fc1(x)
        pred = F.log_softmax(self.fc2(feat))
        # print(pred.size())
        return feat,pred

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters())
softmax_loss_func = nn.NLLLoss()


# for epoch in range(EPOCH):
#     for i,(x,y) in enumerate(train_loader):
#         feat,output = cnn(x)
#         softmax_loss = softmax_loss_func(output,y)
#         data_x = []
#         x_ = feat.data.numpy()
#         # print(b)
#         data_x.append(x_)
#         data_xx = torch.Tensor(data_x).squeeze()
#         # print(a_.size())
#         # b = torch.Tensor([0,1,2,3,4,5,6,7,8,9])
#         center = torch.randn(BATCH_SIZE, 2)
#         center_exp = center.index_select(dim=0, index=y.long())
#         # print(center_exp.size())
#         count = torch.histc(y.data.float(), bins=BATCH_SIZE, min=0, max=50)
#         num = count.index_select(dim=0, index=y.long())
#         center_loss = torch.sum(torch.sqrt(torch.sum((data_xx - center_exp) ** 2, dim=1)) / num)
#         # print(center_loss)
#         # print(softmax_loss)
#         loss = softmax_loss + center_loss
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if i % 100 ==0:
#             print("Epoch:{} step:{}".format(epoch,i))
#             a = []
#             a.append(feat)
#             b = []
#             b.append(y)
#             feat1 = torch.cat(a, 0)
#             labels = torch.cat(b, 0)
#             feats = feat1.data.numpy()
#             label = labels.data.numpy()
#             print(feats)

def visualize(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    print(feat.shape,labels.shape)
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.xlim(xmin=-8,xmax=8)
    plt.ylim(ymin=-8,ymax=8)
    plt.text(-7.8,7.3,"epoch=%d" % epoch)
    plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)

def train(epoch):
    print("Training... Epoch = %d" % epoch)
    ip1_loader = []
    idx_loader = []
    for i,(data, target) in enumerate(train_loader):

        ip1, pred = cnn(data)
        softmax_loss = softmax_loss_func(pred, target)
        data_x = []
        x_ = ip1.data.numpy()
        # print(b)
        data_x.append(x_)
        data_xx = torch.Tensor(data_x).squeeze()
        # print(a_.size())
        # b = torch.Tensor([0,1,2,3,4,5,6,7,8,9])
        center = torch.randn(BATCH_SIZE, 2)
        center_exp = center.index_select(dim=0, index=target.long())
        # print(center_exp.size())
        count = torch.histc(target.data.float(), bins=BATCH_SIZE, min=0, max=50)
        num = count.index_select(dim=0, index=target.long())
        center_loss = torch.sum(torch.sqrt(torch.sum((data_xx - center_exp) ** 2, dim=1)) / num)
        # print(center_loss)
        # print(softmax_loss)
        loss = softmax_loss + center_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ip1_loader.append(ip1)
        idx_loader.append((target))
    # print(ip1_loader,type(ip1_loader))
    # print(idx_loader,type(idx_loader))
    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    print(feat.size(),"\n",labels.size())
    print(feat.data.numpy(),labels.data.numpy())
    visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)

if __name__ == '__main__':

    for epoch in range(100):
        train(epoch+1)

