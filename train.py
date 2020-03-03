# -*- coding: utf-8 -*-
"""
Created on 2019.12.3
@author: junbin
训练
"""
import tensorflow as tf
from data_reload import *
from Network import *
from Vgg_16 import *
import math

LEARNING_RATE_BASE = 0.0001  # 最初学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
LEARNING_RATE_STEP = 1000  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE
batch_size = 128
STEPS = 5000000

def mkdir(name):
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, axis=0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

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


def train():
    datasets, label, val_data, val_label = read_train_data()
    
        global_step =tf.train.get_or_create_global_step()
        tower_grads = []
        with tf.name_scope('Input_data'):
            X = tf.placeholder(tf.float32, [None, 224, 224, 3], name="Input")
            Y = tf.placeholder(tf.float32, [None, 60], name='Estimation')        
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP,LEARNING_RATE_DECAY, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate)
        with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
            for i in range(GPU_num):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                            _x = X[i * batch_size:(i + 1) * batch_size]
                            _y = Y[i * batch_size:(i + 1) * batch_size]
                            logits = ResNet_v2(_x, training=True) 
                            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=logits))
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
                            if i == 0:
                                logits_test = ResNet_v2(_x, training=False)
                                correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): #保证train_op在update_ops执行之后再执行
            grads = average_gradients(tower_grads)
            train_op = opt.apply_gradients(grads, global_step=global_step)
        merged = tf.summary.merge_all()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
        tf_config.gpu_options.allow_growth = True # 自适应显存
        with tf.Session(config=tf_config) as sess:
            with tf.name_scope('init'):
                init_op = tf.global_variables_initializer()
            sess.run(init_op)
            var_list = tf.trainable_variables()
            if global_step is not None:
                var_list.append(global_step)
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
            writer = tf.summary.FileWriter("./logs", sess.graph)
            # 训练模型。
            if tf.train.latest_checkpoint('checkpoint') is not None:
                saver.restore(sess, tf.train.latest_checkpoint('checkpoint'))
            data_len = len(datasets)
            for i in range(STEPS):
                start = (i*batch_size*GPU_num) % int(data_len/(batch_size*GPU_num))*batch_size*GPU_num
                end = start + batch_size*GPU_num
                train_data = datasets[start:end]
                train_label = label[start:end]
                sess.run(train_op, feed_dict={X: train_data, Y: train_label})
                if i % 100 == 0:
                    train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X: train_data, Y: train_label})
                    val_loss, val_acc = sess.run([loss, accuracy], feed_dict={X: val_data, Y: val_label})
                    f = open('./loss.txt', 'a')
                    f.write("%d %g %g %g %g\n" % (i,train_acc,val_acc,train_loss,val_loss))
                    f.close()
                    print("After %g step(s), train accuracy is %g, val accuracy is %g, train loss is %g, val loss is %g" % (i, train_acc,val_acc,train_loss,val_loss))
                if i%1000==0:
                    saver.save(sess, './checkpoint/variable.ckpt', global_step=i)

def main():
    mkdir('./checkpoint')
    train()
    # with tf.name_scope('Input_data'):
    #     X = tf.placeholder(tf.float32, [None, 149, 149, 3], name="Input")
    # logits = ResNet_v2(X, False)
    # merged = tf.summary.merge_all()
    # sess= tf.Session()
    # writer = tf.summary.FileWriter("./logs", sess.graph)

if __name__ == '__main__':
    main()
