# center loss

import tensorflow as tf


def get_center_loss(features, labels, alpha=0.5, num_classes=10):
    '''
    获取center loss及center的更新op
    Arguments:
      features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
      labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
      alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
      num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.

    Return：
      loss: Tensor,可与softmax loss相加作为总的loss进行优化.
      centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
      centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    '''

    # get feature dimension
    len_features = features.get_shape()[1]
    # initailizer class center
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers',
                              [num_classes, len_features],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0),
                              trainable=False)
    # get mini-batch center
    # batchxfeature_len
    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    # compute center loss
    loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(features, centers_batch), axis=-1))
    # loss = tf.nn.l2_loss(features-centers_batch)
    # update center vector
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features
    # count example nums for each class
    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, idx, count = tf.unique_with_counts(labels)
    # get every instance appear times
    appear_times = tf.gather(count, idx)
    appear_times = tf.reshape(appear_times, shape=[-1, 1])
    # delta c
    diff = diff / tf.cast((1 + appear_times), dtype=tf.float32)
    diff = alpha * diff
    # update center
    # centers[labels[i]] - delta_diff[i]
    center_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, center_update_op
