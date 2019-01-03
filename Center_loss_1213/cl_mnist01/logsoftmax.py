import torch
import torch.nn as nn
from torch import optim
import torchvision
from torch.nn import functional as F
import torch.utils.data as data
from matplotlib import pyplot as plt
from cl_mnist02.center_loss import CenterLoss

EPOCH = 1000
BATCH_SIZE = 128
LR = 0.00001


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.PReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.PReLU(),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.PReLU(),
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 2),
            nn.PReLU()
        )
        self.fc2 = nn.Linear(2, 10,bias=False)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = x.view(x.size(0), -1)
        feat = self.fc1(x)
        output = self.fc2(feat)
        # return feat, F.log_softmax(output, dim=1)
        return feat,output

cnn = CNN().cuda()
print(cnn)

# softmax_loss_fn = nn.NLLLoss()
softmax_loss_fn = nn.CrossEntropyLoss()
softmax_opt = optim.Adam(cnn.parameters())

centerloss = CenterLoss(10, 2)
center_opt = optim.Adam(centerloss.parameters())

train_data = torchvision.datasets.MNIST("D:\PycharmProjects\Center_loss_1213\mnist_data", train=True,
                                        transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST("D:\PycharmProjects\Center_loss_1213\mnist_data", train=False,
                                       transform=torchvision.transforms.ToTensor())

train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

if __name__ == '__main__':
    # cnn.load_state_dict(torch.load("centerloss.pkl"))
    for epoch in range(EPOCH):
        features = []
        labels = []
        for step, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()
            feat, output = cnn(img)
            softmax_loss = softmax_loss_fn(output, label)
            center_loss = centerloss(feat, label)
            loss = softmax_loss + center_loss
            print("loss:{0} soft_loss:{1} center_loss:{2}".format(loss.cpu().data.numpy(),
                                                                  softmax_loss.cpu().data.numpy(),
                                                                  center_loss.cpu().data.numpy()))
            softmax_opt.zero_grad()
            center_opt.zero_grad()
            loss.backward()
            softmax_opt.step()
            center_opt.step()

            features.append(feat)
            labels.append(label)

        torch.save(cnn.state_dict(), "param/centerloss.pt")
        features = torch.cat(features).cpu().data.numpy()
        labels = torch.cat(labels).cpu().data.numpy()

        plt.ion()
        plt.clf()
        color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for i in range(10):
            plt.plot(features[i == labels, 0], features[i == labels, 1], ".", c=color[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc="upper right")
        plt.title("epoch%d"%epoch)
        plt.savefig('D:\PycharmProjects\Center_loss_1213\cl_mnist01\softmax_image/epoch=%d.jpg' % (epoch + 1))
        plt.pause(0.1)
    plt.ioff()
    plt.show()
