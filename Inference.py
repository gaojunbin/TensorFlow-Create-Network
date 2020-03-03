# -*- coding: utf-8 -*-
"""
Created on 2019.12.4
@author: junbin

"""

import tensorflow as tf
import numpy as np
from data_reload import *
import seaborn as sns
import math
import matplotlib.pyplot as plt
import pandas as pd
from Network import *

batch_size = 1000
GPU_num = 2

def compute(test_data,test_label, accuracy, sess, X, Y):
    test_accuracy = 0
    num = len(test_data)
    for j in range(0, math.ceil(num / 500)):
        if (j != math.ceil(num / 500) - 1):
            test_accuracy = test_accuracy + accuracy.eval(session=sess,feed_dict={X: test_data[500 * j:500 * j + 500],Y: test_label[500 * j:500 * j + 500]}) * 500
        else:
            test_accuracy = test_accuracy + accuracy.eval(session=sess,feed_dict={X: test_data[500 * j:num],Y: test_label[500 * j:num]}) * (num - 500 * j)
    test_accuracy = test_accuracy / num
    return test_accuracy

def main():
    #sess = tf.Session()
    test_datas, test_labels = read_test_data()
    
    with tf.name_scope('Input_data'):
        X = tf.placeholder(tf.float32, [None, 149, 149, 3], name="Input")
        Y = tf.placeholder(tf.float32, [None, 60], name='Estimation')       

    logits_test = ResNet_v2(X, training=False)
    correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf_config.gpu_options.allow_growth = True # 自适应显存

    with tf.Session(config=tf_config,graph=tf.get_default_graph()) as sess:
    # sess = tf.Session(graph=tf.get_default_graph())
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))

        acc = compute(test_datas,test_labels, accuracy, sess, X, Y)

        print('整体预测准确率：%g'%(acc))

        # hotmap = np.zeros((60,60))
        # prediction_ = sess.run(logits_test, feed_dict={X: test_datas})
        # c = sess.run(tf.argmax(prediction_, 1))#预测
        # h = sess.run(tf.argmax(test_labels, 1))#真实
        # print(h)
        # c = c.astype('int')
        # h = h.astype('int')
        # print(len(c))
        # for i in range(len(c)):
        #     print((h[i],c[i]))
        #     hotmap[h[i]][c[i]] = hotmap[h[i]][c[i]] + 1#列真实，行预测

        # fig, ax = plt.subplots(figsize = (10,10))
        # #sns.heatmap(pd.DataFrame(hotmap, columns = range(0,60), index = range(0,60)),annot=True, vmax=15000,vmin = 0, xticklabels= True, yticklabels= True, square=True)#cmap="YlGnBu"
        # sns.heatmap(hotmap, linewidths=0.02, vmax=30, vmin=0,cmap="YlGnBu")
        # fig.savefig("hotmap.png")
        # plt.show()

if __name__ == '__main__':
    main()
