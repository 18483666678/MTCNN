import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np

from tool import utils

import nets

from torchvision import transforms
import time


class Detector:

    def __init__(self, pnet_param="./param02/pnet.pt", rnet_param="./param02/rnet.pt", onet_param="./param02/onet.pt",
                 isCuda=True):

        self.isCuda = isCuda

        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self, image):

        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        # return pnet_boxes

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        # print( rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset =torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.6)
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.5)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.97)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]


            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        return utils.nms(np.array(boxes), 0.7, isMin=True)

    def __pnet_detect(self, image):

        boxes = []

        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            # print(img_data.size()) #img_data是CHW形状
            img_data.unsqueeze_(0)  #前面加一个维度代表批次 NCHW
            # print(img_data.size())

            _cls, _offset = self.pnet(img_data)
            # print(_cls.size(),"====")
            # print(_offest.size(),"----")

            cls, offset = _cls[0][0].cpu().data, _offset[0].cpu().data

            idxs = torch.nonzero(torch.gt(cls, 0.6))
            # print(cls.size(),"111")
            # print(offset.size(),"222")
            # print(idxs.size(),"333")
            for idx in idxs:
                #大于0.6所有的框添加进boxes
                boxes.append(self.__box(idx, offset, cls[idx[0], idx[1]], scale))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        return utils.nms(np.array(boxes), 0.5)  #阈值0.5

    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        #特征图的索引是HW先高后宽
        _x1 = (start_index[1].data.float() * stride) / scale
        _y1 = (start_index[0].data.float() * stride) / scale
        _x2 = (start_index[1].data.float() * stride + side_len) / scale
        _y2 = (start_index[0].data.float() * stride + side_len) / scale
        # print(cls,"cls")
        # print(start_index,"idx")
        # print(start_index[0],"\n",start_index[1])
        # print(offset.size,"offset")

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        # print(offset.size(),"555")
        # print(_offset.size(),"4444")
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]


if __name__ == '__main__':

    image_file = r"D:\image_face\17.jpg"
    # image_file = r"D:\image_face\1.jpg"

    detector = Detector()

    with Image.open(image_file) as im:

        boxes = detector.detect(im)
        print(im.size)
        imDraw = ImageDraw.Draw(im)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            print(box[4])
            imDraw.rectangle((x1, y1, x2, y2), outline='red',width=3)

        im.show()
