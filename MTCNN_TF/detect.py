import os
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from tool import utils
import net


class Detector:
    def __init__(self, pnet_param="./tfparam/12", rnet_param="./tfparam/24", onet_param="./tfparam/48"):

        self.pnet = nets_tf.Net(12)
        self.pnet.load_param(pnet_param)

        self.rnet = nets_tf.Net(24)
        self.rnet.load_param(rnet_param)

        self.onet = nets_tf.Net(48)
        self.onet.load_param(onet_param)

    def detect(self, image):

        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        # return pnet_boxes

        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        # return rnet_boxes

        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])

        return onet_boxes

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
            img_data = np.array(img) / 255. - 0.5
            # print(img_data.shape)
            _img_dataset.append(img_data)

        img_dataset = np.stack(_img_dataset)
        # print(img_dataset.shape)

        cls, offset = self.onet.sess.run([self.onet.cls_pre, self.onet.off_pre],
                                         feed_dict={self.onet.input: img_dataset})
        cls = np.reshape(cls, (-1, 1))
        offset = np.reshape(offset, (-1, 4))
        # print("debug", cls, offset)
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

            img_data = np.array(img) / 255. - 0.5
            # print(img_data.shape)
            _img_dataset.append(img_data)

        img_dataset = np.stack(_img_dataset)
        # print(img_dataset.shape)

        cls, offset = self.rnet.sess.run([self.rnet.cls_pre, self.rnet.off_pre],
                                         feed_dict={self.rnet.input: img_dataset})
        # print("debug", cls, offset)
        boxes = []
        idxs, _ = np.where(cls > 0.7)
        # print(idxs, idxs.shape)
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

        return utils.nms(np.array(boxes), 0.5, isMin=True)

    def __pnet_detect(self, image):

        boxes = []
        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1
        while min_side_len > 12:
            img_data = np.array(img) / 255. - 0.5
            img_data = img_data[np.newaxis, :]
            # print(img_data, img_data.shape)

            _cls, _offest = self.pnet.sess.run([self.pnet.cls_pre, self.pnet.off_pre],
                                               feed_dict={self.pnet.input: img_data})
            # print(_cls, _cls.shape)
            # print(_offest, _offest.shape)

            cls, offest = _cls[0, :, :, 0], _offest[0]
            # print(cls, cls.shape)
            # print(offest, offest.shape)

            idxs = np.where(cls > 0.6)
            # print(np.stack(idxs, axis=1))
            # print(list(zip(idxs[0], idxs[1])))

            for idx in list(zip(idxs[0], idxs[1])):
                # print("debug", idx[0], idx[1])
                # print(cls[idx[0], idx[1]])
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        return utils.nms(np.array(boxes), 0.5)

    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = (start_index[1] * stride) / scale
        _y1 = (start_index[0] * stride) / scale
        _x2 = (start_index[1] * stride + side_len) / scale
        _y2 = (start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[start_index[0], start_index[1], :]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]


save_path_base = "./tfparam"
dataset_path_base = "../samples/"
size = 12
save_path = os.path.join(save_path_base, str(size))
dataset_path = os.path.join(dataset_path_base, str(size))

if __name__ == '__main__':
    # image_file = "../../img_celeba_4dbg/24/positive/0.jpg"
    image_file = "../img_celeba_4dbg/000002.jpg"
    detector = Detector()

    with Image.open(image_file) as im:
        # im.show()
        boxes = detector.detect(im)
        print(im.size)
        imDraw = ImageDraw.Draw(im)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            print(box[4])
            imDraw.rectangle((x1, y1, x2, y2), outline='red')

        im.show()