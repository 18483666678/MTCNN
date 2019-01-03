import tensorflow as tf
import os
import numpy as np

from PIL import Image


# 封装数据集
class FaceDataset:
    def __init__(self, path):
        self.path = path  # 路径
        self.data = []  # 存储列表
        # 按行读取 extend添加进dataset列表里
        self.data.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.data.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.data.extend(open(os.path.join(path, "part.txt")).readlines())
        np.random.shuffle(self.data)

        self.dataset = tf.data.Dataset.from_tensor_slices(self.data)  # 变成tensor
        # py_func(func, inp, Tout, stateful=True, name=None)  func是自定义函数，
        # inp是输入到自定义函数的参数，Tout是自定义函数返回的数据类型
        self.dataset = self.dataset.map(
            lambda data: tuple(tf.py_func(self.read_data, [data], [tf.float32, tf.float32, tf.float32])))

        self.dataset = self.dataset.shuffle(buffer_size=500)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(512)

        self.iterator = self.dataset.make_one_shot_iterator()
        self.iterator = self.iterator.get_next()

    def get_batch(self, sess):
        return sess.run(self.iterator)

    def read_data(self, data):  # 这是Dataset里定义的  index处理那条数据
        # "{0}.jpg 置信度 x1 y1 x2 y2 "
        data = data.decode()  # 去空格和换行符 用空格分割
        strs = data.strip(" ").split()
        img_path = os.path.join(self.path, strs[0])
        cond = np.array([float(strs[1])], dtype=np.float32)  # 置信度
        offset = np.array([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])], dtype=np.float32)
        img_data = np.array(Image.open(img_path),
                            dtype=np.float32) / 255. - 0.5  # 数据做归一化转换成tensor类型                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      )

        return img_data, cond, offset  # 返回的顺序是 数据，置信度，偏移量


if __name__ == '__main__':
    dataset = FaceDataset(r"E:\celeba1208\12")
    with tf.Session() as sess:
        p_img_data, p_cond, p_offset = dataset.get_batch(sess)
        # print(p_img_data, p_cond, "111111\n", p_offset)
        print(p_cond.shape)
        print(p_offset.shape)
        print(p_img_data.shape)
