from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image

#封装数据集
class FaceDataset(Dataset):  # 继承Dataset
    def __init__(self, path):
        self.path = path  # 路径
        self.dataset = []  # 存储列表
        # 按行读取 extend添加进dataset列表里
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __getitem__(self, index):  # 这是Dataset里定义的  index处理那条数据
        # "{0}.jpg 置信度 x1 y1 x2 y2 "
        strs = self.dataset[index].strip().split(" ")  # 去空格和换行符 用空格分割
        img_path = os.path.join(self.path,strs[0])
        cond = torch.Tensor([int(strs[1])])  # 置信度
        offset = torch.Tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])
        img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5) #数据做归一化转换成tensor类型
        # print(img_data.shape,"imgdata")
        img_data = img_data.permute(2,0,1)
        # img_data = img_data.transpose(2,0)
        # print(img_data.shape,"123")
        # img_data = img_data.transpose(2,1)
        # print(img_data.shape,"444")
        # a = img_data.permute(2,0,1) #轴交换 转置
        # print(a.shape                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            )

        return img_data,cond,offset #返回的顺序是 数据，置信度，偏移量

    def __len__(self): #也是Dataset的
        return len(self.dataset)


if __name__ == '__main__':
    dataset = FaceDataset(r"D:\celeba3\12")
    # print(dataset[1])
