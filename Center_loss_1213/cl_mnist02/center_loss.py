import torch
import torch.nn as nn


#欧式距离
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, isCuda=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.isCuda = isCuda

        if isCuda:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward(self, feat, label):
        centers_exp = self.centers.index_select(0, label)
        count = torch.histc(label.cpu().data.float(), bins=self.num_classes, min=0, max=self.num_classes)
        if self.isCuda:
            count = count.cuda()
        num = count.index_select(dim=0, index=label)
        loss = torch.sum(torch.sqrt(torch.sum((feat - centers_exp) ** 2, dim=1)) / num) / label.size(0)
        # loss = (torch.sqrt((feature - center_exp).pow(2).sum(1) / count)).sum() / label.size(0) #同上
        # loss = (torch.sqrt((feat-centers_exp).pow(2).sum(1)/num)).sum()/label.size(0)
        return loss


#余弦相似度
import numpy as np
class CenterLoss2(nn.Module):
    def __init__(self,num_classes,feature_dim,isCuda=True):
        super(CenterLoss2,self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.isCuda = isCuda
        if isCuda:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward(self, feat,label):
        f_feat = torch.norm(feat,2,self.num_classes,True)
        count = f_feat.expand_as(label)
        f_center = torch.norm(label,2,self.num_classes,True)
        center = f_center.expand_as(label)
        l = feat.div(f_feat)
        e = center.div(f_center)
        m = np.dot(l,e[0])
        n = np.dot(l.numpy(),e[1].numpy())
        loss = torch.sum(count*center) / (torch.sqrt(torch.sum(count**2)) * torch.sqrt(torch.sum(center**2)))
        return loss