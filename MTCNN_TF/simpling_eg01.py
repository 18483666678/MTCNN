import tensorflow as tf
import numpy as np
import cv2
import os
import random
from PIL import Image


class Sample:

    def __init__(self,path):

        self.path=path

        self.data=[]
        self.data.extend(open(os.path.join(self.path,r"positive.txt")).readlines())
        self.data.extend(open(os.path.join(self.path,r"part.txt")).readlines())
        self.data.extend(open(os.path.join(self.path,r"negative.txt")).readlines())
        np.random.shuffle(self.data)

        self.dataset=tf.data.Dataset.from_tensor_slices(self.data)
        self.dataset=self.dataset.map(lambda data:tuple(tf.py_func(self.read_data,[data],[tf.float32,tf.float32,tf.float32,tf.float32])))

        self.dataset=self.dataset.shuffle(buffer_size=500)
        self.dataset=self.dataset.repeat()
        self.dataset=self.dataset.batch(512)

        self.iterator=self.dataset.make_one_shot_iterator()
        self.itemlent=self.iterator.get_next()

    def get_batch(self,sess):
        return sess.run(self.itemlent)

    def read_data(self,data):
        data=data.decode()
        strs=data.strip(" ").split()
        img_path=os.path.join(self.path,strs[0])
        img_data=np.array(Image.open(img_path),dtype=np.float32)/255-0.5
        conf=np.array([float(strs[1])],dtype=np.float32)
        off=np.array([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])],dtype=np.float32)
        landmark=np.array([float(strs[6]),float(strs[7]),float(strs[8]),float(strs[9]),float(strs[10]),float(strs[11]),float(strs[12]),float(strs[13]),float(strs[14]),float(strs[15])],dtype=np.float32)

        return img_data,conf,off,landmark

if __name__ == '__main__':
    rnet_simple = Sample(r"D:\样本一\12")
    with tf.Session() as sess:

        rnet_img_data, rnet_conf, rnet_off ,rnet_landmark= rnet_simple.get_batch(sess)

        print(rnet_conf)