import tensorflow as tf
from simpling_eg02 import FaceDataset
import numpy as np

save_path = r"D:\新建文件夹\MTCNN_TF\param"


class PNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 12, 12, 3])
        self.cond = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.offset = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    def forward(self):
        self.conv1 = tf.nn.leaky_relu(tf.layers.conv2d(self.x, 10, 3))
        self.pool1 = tf.layers.max_pooling2d(self.conv1, 2, (2, 2))
        self.conv2 = tf.nn.leaky_relu(tf.layers.conv2d(self.pool1, 16, 3))
        self.conv3 = tf.nn.leaky_relu(tf.layers.conv2d(self.conv2, 32, 3))

        self.conv4_1 = tf.nn.sigmoid(tf.layers.conv2d(self.conv3, 1, 1))
        self.out_cond = tf.reshape(self.conv4_1, shape=[-1, 1])
        # print(self.out_cond,self.out_cond.shape)

        self.conv4_2 = tf.layers.conv2d(self.conv3, 4, 1)
        self.out_offset = tf.reshape(self.conv4_2, shape=[-1, 4])
        # print(self.out_offset,self.out_offset.shape)
        print("0",self.out_offset.shape)

    def backward(self):
        # cls_mask = tf.where(self.cond < 2)
        # self.cls_l = tf.gather(self.cond, cls_mask)[:, 0]   #tf.gather根据索引，从输入张量中依次取元素，构成一个新的张量。
        # self.cls_p = tf.gather(self.out_cond, cls_mask)[:, 0]
        #
        # off_mask = tf.where(self.cond > 0)
        # self.off_l = tf.gather(self.offset, off_mask)[:, 0]
        # self.off_p = tf.gather(self.out_offset, off_mask)[:, 0]

        self.cond_mask = tf.less(self.cond, 2)
        self.cond_index = tf.where(tf.equal(self.cond_mask, True))[:, 0]
        self.cond_label = tf.gather(self.cond, self.cond_index)
        self.cond_pred = tf.gather(self.out_cond, self.cond_index)

        self.offset_mask = tf.greater(self.cond, 0)
        print("1",self.offset_mask.shape)
        self.offset_index = tf.where(tf.equal(self.offset_mask, True))[:, 0]
        print("2",self.offset_index.shape)
        self.offset_lebal = tf.gather(self.offset, self.offset_index)
        print("3",self.offset_lebal.shape)
        self.offset_pred = tf.gather(self.out_offset, self.offset_index)
        print("4",self.offset_pred.shape)
        self.cond_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.cond_label, logits=self.cond_pred))
        self.offset_loss = tf.reduce_mean(tf.square(self.offset_lebal - self.offset_pred))

        self.loss = self.cond_loss + self.offset_loss
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # cond_mask = tf.less(self.cond,2)
        # cond = tf.boolean_mask(self.cond,cond_mask)
        # out_cond = tf.boolean_mask(self.out_cond,cond_mask)
        # print(out_cond,out_cond.shape)
        # print(cond,cond.shape)
        # self.cond_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=cond, logits=out_cond))

        # offset_mask = tf.greater(self.cond,0)
        # offset_index = tf.count_nonzero(offset_mask)
        # offset = self.offset[offset_index]
        # out_offset = self.out_offset[offset_index]
        # self.offset_loss = tf.reduce_mean(tf.square(out_offset - offset))
        # self.loss = self.cond_loss + self.offset_loss
        # self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    net = PNet()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()

    f_data = FaceDataset(r"E:\celeba1208\12")

    save = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # save.restore(sess,save_path)
        for epoch in range(100):
            img_data, cond, offset = f_data.get_batch(sess)

            for i in range(1000000):
                loss, cond_loss, offset_loss, _ = sess.run(
                    [net.loss, net.cond_loss, net.offset_loss, net.optimizer], feed_dict={
                        net.x: img_data, net.cond: cond, net.offset: offset})
                if i % 1000 == 0:
                    print("批次：",epoch,"loss:", loss, "cond_loss:", cond_loss, "offset_loss:", offset_loss)
                    save.save(sess, save_path)
