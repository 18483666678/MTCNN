import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CenterLoss(nn.Module):

    def __init__(self, num_classes, feature_dim, use_cuda=False):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward(self, feature, label):
        softmaxcenter = self.centers.index_select(0, label)
        if self.use_cuda:
            hist = torch.histc(label.cpu().float(), bins=self.num_classes, min=0, max=self.num_classes).cuda()
        else:
            hist = torch.histc(label.float(), bins=self.num_classes, min=0, max=self.num_classes)
        count = hist.index_select(0, label)
        loss = (torch.sqrt((feature - softmaxcenter).pow(2).sum(1) / count)).sum() / label.size(0)
        return loss


class CNNNet(nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, 2)
        )
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size(0),-1)
        # y = y.view(-1, 32 * 7 * 7)
        feature = self.fc(y)
        y = self.fc2(feature)
        return feature, F.log_softmax(y, 1)


transform = transforms.Compose([
    transforms.ToTensor(),
])
train_data = torchvision.datasets.MNIST("MNIST_data/", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

test_data = torchvision.datasets.MNIST("MNIST_data/", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)

if __name__ == '__main__':
    cnn = CNNNet().cuda()
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(cnn.parameters(), lr=0.0001)
    center_loss = CenterLoss(10, 2, use_cuda=True)
    opt_cent = optim.Adam(center_loss.parameters(), lr=0.0001)

    # cnn.load_state_dict(torch.load("centerlossdemo.pkl"))

    for epoch in range(100):


        features = []
        labels = []
        # Train stage
        for step, (img, label) in enumerate(train_loader):
            img = img.to("cuda")
            label = label.to("cuda")
            feature, out = cnn(img)
            # pred = torch.max(out, 1)[1]
            # print("predicted", predicted, predicted.shape)
            # print(label.size(0))
            # print("Accuracy", (predicted == label).sum().float() / float(label.size(0)))

            softmaxloss = loss_func(out, label)
            centerloss = center_loss(feature, label)
            loss = softmaxloss + centerloss
            print("loss:",loss,"softmaxloss:", softmaxloss, "  centerloss:", centerloss)
            opt.zero_grad()
            opt_cent.zero_grad()
            loss.backward()
            opt.step()
            opt_cent.step()
            features.append(feature)
            labels.append(label)

        torch.save(cnn.state_dict(), "centerloss.pkl")
        features = torch.cat(features).cpu().data.numpy()
        labels = torch.cat(labels).cpu().data.numpy()
        plt.ion()
        plt.clf()
        # fig = plt.figure()
        # ax = Axes3D(fig)
        color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for i in range(10):
            plt.plot(features[i == labels, 0], features[i == labels, 1], ".", c=color[i])
            # ax.scatter(features[i == labels, 0], features[i == labels, 1], features[i == labels, 2], c=color(i),
            #            marker='.', s=50, label='')
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc="upper right")
        plt.xlim(xmin=-8, xmax=8)
        plt.ylim(ymin=-8, ymax=8)
        plt.text(-7.8, 7.3, "epoch=%d" % epoch)
        plt.savefig('./images/epoch=%d.jpg' % epoch)
        plt.pause(1)
    plt.ioff()
    plt.show()

        # # Test stage
        # with torch.no_grad():
        #     for img, label in test_loader:
        #         out = cnn(img)
        #         predict = torch.max(out, 1)[1]
        #         print("Accuracy", (predict == label).sum().float() / float(label.size(0)))