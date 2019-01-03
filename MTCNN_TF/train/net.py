import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import layers as nn
from torch.utils.data import DataLoader
from simpling_eg02 import FaceDataset

class PNet:

    def __init__(self, scope):
        self.scope = scope
        print("Init PNet.")
        # 1. Input:
        # self.x = tf.placeholder(tf.float32, shape=[None, 12, 12, 3])

    def forward(self, x):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # 2. Common Networks Layers
            self.conv1 = nn.conv2d(inputs=x, filters=10, kernel_size=[3, 3], activation=tf.nn.relu)
            self.conv1 = nn.max_pooling2d(self.conv1, pool_size=2, strides=2)
            self.conv2 = nn.conv2d(inputs=self.conv1, filters=16, kernel_size=[3, 3], activation=tf.nn.relu)
            self.conv3 = nn.conv2d(inputs=self.conv2, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)

            self.conv4_1 = nn.conv2d(inputs=self.conv3, filters=16, kernel_size=[1, 1], activation=tf.nn.relu)
            self.conv4_1 = nn.conv2d(inputs=self.conv4_1, filters=8, kernel_size=[1, 1], activation=tf.nn.relu)
            self.conv4_1 = nn.conv2d(inputs=self.conv4_1, filters=1, kernel_size=[1, 1], activation=tf.nn.sigmoid)

            self.conv4_2 = nn.conv2d(inputs=self.conv3, filters=4, kernel_size=[1, 1])

        return self.conv4_1, self.conv4_2


class RNet:

    def __init__(self, scope):
        self.scope = scope
        print("Init RNet.")
        # 1. Input:
        # self.x = tf.placeholder(tf.float32, shape=[None, 24, 24, 3])

    def forward(self, x):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # 2. Common Networks Layers
            self.conv1 = nn.conv2d(inputs=x, filters=28, kernel_size=[3, 3], activation=tf.nn.relu)
            self.conv1 = nn.max_pooling2d(self.conv1, pool_size=3, strides=2)
            self.conv2 = nn.conv2d(inputs=self.conv1, filters=48, kernel_size=[3, 3], activation=tf.nn.relu)
            self.conv2 = nn.max_pooling2d(self.conv2, pool_size=3, strides=2)
            self.conv3 = nn.conv2d(inputs=self.conv2, filters=64, kernel_size=[2, 2], activation=tf.nn.relu)
            # self.conv4 = nn.conv2d(inputs=self.conv3, filters=64, kernel_size=[2, 2], activation=tf.nn.relu)
            # self.conv5_1 = nn.conv2d(inputs=self.conv4, filters=1, kernel_size=[1, 1], activation=tf.nn.sigmoid)
            # self.conv5_2 = nn.conv2d(inputs=self.conv4, filters=4, kernel_size=[1, 1])
            #
            # return self.conv5_1, self.conv5_2

            # This is a wrong method, because the batch number is unknown
            # self.conv_flat = tf.reshape(self.conv3, [self.conv3.get_shape()[0], -1])
            self.conv_flat = tf.reshape(self.conv3, (-1, 2 * 2 * 64))
            self.fc1 = tf.layers.dense(inputs=self.conv_flat, units=128, activation=tf.nn.relu)
            self.fc1 = tf.layers.dense(inputs=self.fc1, units=64, activation=tf.nn.relu)
            self.fc1 = tf.layers.dense(inputs=self.fc1, units=32, activation=tf.nn.relu)
            self.fc2_1 = tf.layers.dense(inputs=self.fc1, units=1, activation=tf.nn.sigmoid)
            self.fc2_2 = tf.layers.dense(inputs=self.fc1, units=4)

        return self.fc2_1, self.fc2_2


class ONet:

    def __init__(self, scope):
        self.scope = scope
        print("Init ONet.")
        # 1. Input:
        # self.x = tf.placeholder(tf.float32, shape=[None, 48, 48, 3])

    def forward(self, x):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # 2. Common Networks Layers
            self.conv1 = nn.conv2d(inputs=x, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
            self.conv1 = nn.max_pooling2d(self.conv1, pool_size=3, strides=2)
            self.conv2 = nn.conv2d(inputs=self.conv1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
            self.conv2 = nn.max_pooling2d(self.conv2, pool_size=3, strides=2)
            self.conv3 = nn.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
            self.conv3 = nn.max_pooling2d(self.conv3, pool_size=2, strides=2)
            self.conv4 = nn.conv2d(inputs=self.conv3, filters=128, kernel_size=[2, 2], activation=tf.nn.relu)
            self.conv5 = nn.conv2d(inputs=self.conv4, filters=64, kernel_size=[2, 2], activation=tf.nn.relu)
            self.conv6_1 = nn.conv2d(inputs=self.conv5, filters=1, kernel_size=[1, 1], activation=tf.nn.sigmoid)
            self.conv6_2 = nn.conv2d(inputs=self.conv5, filters=4, kernel_size=[1, 1])

        return self.conv6_1, self.conv6_2


class Net:

    def __init__(self, net_size=12):
        self.input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.cls = tf.placeholder(tf.float32, shape=[None, 1])
        self.off = tf.placeholder(tf.float32, shape=[None, 4])

        if net_size == 12:
            self.scope = "pnet"
            self.net = PNet(self.scope)
        elif net_size == 24:
            self.scope = "rnet"
            self.net = RNet(self.scope)
        elif net_size == 48:
            self.scope = "onet"
            self.net = ONet(self.scope)
        else:
            print("The vaild net size is (12, 24, 48)!")
            exit(-1)

        self.forword()
        self.backward()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver([v for v in tf.global_variables() if v.name[0:4] == self.scope])

    def forword(self):
        self.cls_pre, self.off_pre = self.net.forward(self.input)
        self.cls_p = tf.reshape(self.cls_pre, (-1, 1))
        self.off_p = tf.reshape(self.off_pre, (-1, 4))

        cls_mask = tf.where(self.cls < 2)
        self.cls_l = tf.gather(self.cls, cls_mask)[:, 0]
        self.cls_p = tf.gather(self.cls_p, cls_mask)[:, 0]

        # cls_mask = tf.where(tf.equal(self.cls, 0))
        # self.cls_l_n = tf.gather(self.cls, cls_mask)[:, 0]
        # self.cls_p_n = tf.gather(self.cls_p, cls_mask)[:, 0]
        #
        # cls_mask = tf.where(tf.equal(self.cls, 1))
        # self.cls_l_p = tf.gather(self.cls, cls_mask)[:, 0]
        # self.cls_p_p = tf.gather(self.cls_p, cls_mask)[:, 0]

        off_mask = tf.where(self.cls > 0)
        self.off_l = tf.gather(self.off, off_mask)[:, 0]
        self.off_p = tf.gather(self.off_p, off_mask)[:, 0]

    def backward(self):
        self.cls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.cls_l, logits=self.cls_p))
        # self.cls_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.cls_l_p, logits=self.cls_p_p)) \
        #                 + tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.cls_l_n, logits=self.cls_p_n))
        self.off_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.off_l, self.off_p))

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.cls_opt = tf.train.AdamOptimizer(0.0002).minimize(self.cls_loss)
            self.off_opt = tf.train.AdamOptimizer().minimize(self.off_loss)

    def load_param(self, param_path):
        checkpoint = tf.train.get_checkpoint_state(param_path)
        print("zeng>> ck", checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights from", param_path)


save_path_base = "D:\新建文件夹\MTCNN_TF\param"
dataset_path_base = r"E:\celeba1208"
size = 12      #,24,48
save_path = os.path.join(save_path_base, str(size))
dataset_path = os.path.join(dataset_path_base, str(size))

if __name__ == '__main__':
    net = Net(size)

    faceDataset = FaceDataset(dataset_path)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        print(tf.global_variables())

        checkpoint = tf.train.get_checkpoint_state(save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        for epoch in range(1000000):
            print(epoch)
            _img_data, _category, _offset = faceDataset.get_batch(sess)
            img_data = _img_data
            category = _category
            offset = _offset
            # print(category)
            # print(offset)

            cls, _, off, _ = sess.run([net.cls_loss, net.cls_opt, net.off_loss, net.off_opt],
                                      feed_dict={net.input: img_data, net.cls: category, net.off: offset})

            if (epoch) % 10 == 0 :
                print(cls, off)
                # cls, cls_p, off, off_p = sess.run([net.cls_l, net.cls_p, net.off_l, net.off_p],
                #                                   feed_dict={net.input: img_data, net.cls: category,
                #                                              net.off: offset})
                # print("zeng>>\n", cls, off)
                # print("zeng>>\n", cls_p, off_p)
                # time.sleep(5)
                saver.save(sess, os.path.join(save_path, "mtcnn"), global_step=epoch)