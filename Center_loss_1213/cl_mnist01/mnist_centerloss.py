import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import numpy as np



EPOCH = 100
BATCH_SIZE = 128

train_data = torchvision.datasets.MNIST("mnist_data/",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST("mnist_data/",train=False,transform=torchvision.transforms.ToTensor())

# train_loader = data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
#
# test_x = torch.autograd.Variable(torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000]/255.)
# test_y = test_data.test_labels[:2000]
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

class CenterLoss(nn.Module):
    def __init__(self,num_classes,feature_dim,isCuda=False):
        super(CenterLoss,self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.isCuda = isCuda

        if self.isCuda:
            #随机正太分布center
            self.centers = nn.Parameter(torch.randn(self.num_classes,self.feature_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes,self.feature_dim))

    def forward(self, feature,label): #label类别，feature特征数据
        center_exp = self.centers.index_select(0,label) #扩充后的center dim=0
        if self.isCuda:
            #统计每个类别的数量
            count = torch.histc(label.cpu().float(),bins=self.num_classes,min=0,max=self.num_classes).cuda()
        else:
            count = torch.histc(label.cpu().float(),bins=self.num_classes,min=0,max=self.num_classes)
        num = count.index_select(dim=0,index=label) #扩充维度符合centers的维度
        loss = torch.sum(torch.sqrt(torch.sum((feature-center_exp)**2,dim=1))/num) / label.size(0)
        # loss = (torch.sqrt((feature - center_exp).pow(2).sum(1) / count)).sum() / label.size(0) #同上
        return loss


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.pre_layer = nn.Sequential(
#             nn.Conv2d(1,16,5,1,2),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16,32,5,1,2),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc1 = nn.Linear(32*7*7,2)
#         self.fc2 = nn.Linear(2,10)
#
#     def forward(self, x):
#         x = self.pre_layer(x)
#         x = x.view(x.size(0),-1)
#         feat = self.fc1(x)
#         pred = F.log_softmax(self.fc2(feat))
#         # print(pred.size())
#         return feat,pred

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*3*3, 2)
        self.ip2 = nn.Linear(2, 10, bias=False)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*3*3)
        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        print("ip2:",ip2)
        return ip1, F.log_softmax(ip2, dim=1)

if __name__ == '__main__':

    cnn = CNN().cuda()
    print(cnn)

    softmax_loss_func = nn.NLLLoss()
    optimizer_softmax = torch.optim.Adam(cnn.parameters(),lr=1e-4)
    centerloss = CenterLoss(10,2,isCuda=True)
    optimizer_center = torch.optim.Adam(centerloss.parameters(),lr=1e-4)  #lr=0.0001
    cnn.load_state_dict(torch.load("centerloss.pkl"))
    for epoch in range(EPOCH):

        print("epoch:{}".format(epoch+1),"***************************")
        features = []
        labels = []

        #Train stage
        for step,(img,label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()
            feature,output = cnn(img)
            #求精度
            # pred = torch.max(out, 1)[1]
            # print("predicted", predicted, predicted.shape)
            # print(label.size(0))
            # print("Accuracy", (predicted == label).sum().float() / float(label.size(0)))

            #softmax+center损失
            softmax_loss = softmax_loss_func(output,label)
            center_loss = centerloss(feature,label)
            loss = softmax_loss + center_loss
            print("loss:",loss,"softmaxloss:",softmax_loss,"centerloss:",center_loss)

            optimizer_softmax.zero_grad()
            optimizer_center.zero_grad()
            loss.backward()
            optimizer_softmax.step()
            optimizer_center.step()

            features.append(feature)
            labels.append(label)
        # torch.save(cnn.state_dict(),"centerloss.pkl")

        features = torch.cat(features).cpu().data.numpy()
        labels = torch.cat(labels).cpu().data.numpy()

        plt.ion()
        plt.clf()
        color = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        for i in range(10):
            plt.plot(features[i == labels,0],features[i == labels,1],".",c=color[i])
        plt.legend(['0','1','2','3','4','5','6','7','8','9'],loc="upper right")
        plt.xlim(xmin=-8, xmax=8)
        plt.ylim(ymin=-8, ymax=8)
        plt.text(-7.8, 7.3, "epoch=%d" % epoch)
        plt.savefig('./images02/epoch=%d.jpg' % epoch)
        plt.pause(1)
    plt.ioff()
    plt.show()