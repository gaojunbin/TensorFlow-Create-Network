# -*- coding: utf-8 -*-
"""
Created on 2019.12.4
@author: junbin
The ResNet_v2 networks
"""

import tensorflow as tf

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape,mean = 0.0, stddev=0.01)
    return tf.Variable(initial, name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def conv2d(input, filter, strides=[1,1,1,1], padding="SAME", name=None):
    # filters with shape [filter_height * filter_width * in_channels, output_channels]
    # Must have strides[0] = strides[3] =1
    # For the most common case of the same horizontal and vertices strides, strides = [1, stride, stride, 1]
    '''
    Args:
        input: A Tensor. Must be one of the following types: float32, float64.
        filter: A Tensor. Must have the same type as input.
        strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input.
        padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
        use_cudnn_on_gpu: An optional bool. Defaults to True.
        name: A name for the operation (optional).
    '''
    return tf.nn.conv2d(input, filter, strides, padding=padding, name=name)  # padding="SAME"用零填充边界

def max_pool_2x2(input, name):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name=name)

def Conv(input, name, filter_size, bias_size, stride, padding = 'SAME'):
    with tf.name_scope(name):
        with tf.name_scope('Variable'):
            filters = weight_variable(filter_size, name='filter')
            bias = weight_variable(bias_size, name='bias')
        with tf.name_scope("Convolution"):
            layer = conv2d(input, filters, strides=stride, padding = padding) + bias
    return layer

def original_residual(block_input, name, block_num=2, size='SAME'):
    '''
    block_input:    代表输入的张量
    block_num:      代表内部卷积次数
    size:           SAME代表输入张量形态不变，反之size缩小一半，channel扩大一倍
    '''
    with tf.name_scope(name):
        net = block_input
        channel = block_input.get_shape().as_list()[-1]
        if size=='SAME':
            for i in range(block_num):
                net = Conv(net, name='conv'+str(i+1), filter_size=[3,3,channel,channel], bias_size=[channel], stride=[1,1,1,1])
                if i != block_num-1:
                    net = tf.nn.relu(net)
            net = tf.nn.relu(net + block_input)
        elif size=='VALID':
            net = tf.nn.relu(Conv(net, name='conv1', filter_size=[3,3,channel,2*channel], bias_size=[2*channel], stride=[1,2,2,1]))
            for i in range(1, block_num):
                net = Conv(net, name='conv'+str(i+1), filter_size=[3,3,2*channel,2*channel], bias_size=[2*channel], stride=[1,1,1,1])
                if i != block_num-1:
                    net = tf.nn.relu(net)
            block_input = Conv(block_input, name='shortcut', filter_size=[1,1,channel,2*channel], bias_size=[2*channel], stride=[1,2,2,1], padding = 'SAME')
            net = tf.nn.relu(net+block_input)
    return net

# def proposed_residual(block_input, name, block_num=2, size='SAME', is_training=False):
#     '''
#     block_input:    代表输入的张量
#     block_num:      代表内部卷积次数
#     size:           SAME代表输入张量形态不变，反之size缩小一半，channel扩大一倍
#     '''
#     with tf.name_scope(name):
#         net = block_input
#         channel = block_input.get_shape().as_list()[-1]
#         if size=='SAME':
#             for i in range(block_num):
#                 net = tf.layers.batch_normalization(net, training=is_training)
#                 net = tf.nn.relu(net)
#                 net = Conv(net, name='conv'+str(i+1), filter_size=[3,3,channel,channel], bias_size=[channel], stride=[1,1,1,1])
#             net = net + block_input
#         elif size=='VALID':
#             net = tf.layers.batch_normalization(net, training=is_training)
#             net = Conv(tf.nn.relu(net), name='conv1', filter_size=[3,3,channel,2*channel], bias_size=[2*channel], stride=[1,2,2,1])
#             for i in range(1, block_num):
#                 net = tf.layers.batch_normalization(net, training=is_training)
#                 net = tf.nn.relu(net)
#                 net = Conv(net, name='conv'+str(i+1), filter_size=[3,3,2*channel,2*channel], bias_size=[2*channel], stride=[1,1,1,1])
#             block_input = Conv(block_input, name='shortcut', filter_size=[1,1,channel,2*channel], bias_size=[2*channel], stride=[1,2,2,1], padding = 'SAME')
#             net = net + block_input
#         return net

def proposed_residual(net, name, block_num=2, size='SAME', is_training=False):
    '''
    block_input:    代表输入的张量
    block_num:      代表内部卷积次数
    size:           SAME代表输入张量形态不变，反之size缩小一半，channel扩大一倍
    '''
    block_input = net
    channel = block_input.get_shape().as_list()[-1]
    with tf.name_scope(name):
        if size=='SAME':
            for i in range(block_num):
                net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope=name+'_BN'+str(i))
                net = tf.nn.relu(net,name = name+'_activation'+str(i))
                net = tf.contrib.layers.conv2d(net,num_outputs=channel,kernel_size=(3,3),stride=(1,1),padding='SAME',activation_fn=None, scope=name+'_Conv'+str(i))
            net = net + block_input
        elif size=='VALID':
            net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope=name+'_BN0')
            net = tf.nn.relu(net,name = name+'_activation0')
            net = tf.contrib.layers.conv2d(net,num_outputs=channel*2,kernel_size=(3,3),stride=(2,2),padding='SAME',activation_fn=None, scope=name+'_Conv0')
            for i in range(1, block_num):
                net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope=name+'_BN'+str(i))
                net = tf.nn.relu(net,name = name+'_activation'+str(i))
                net = tf.contrib.layers.conv2d(net,num_outputs=channel*2,kernel_size=(3,3),stride=(1,1),padding='SAME',activation_fn=None, scope=name+'_Conv'+str(i))
            block_input = tf.contrib.layers.conv2d(block_input,num_outputs=channel*2,kernel_size=(1,1),stride=(2,2),padding='SAME',activation_fn=None, scope=name+'_shortcut')
            net = net + block_input
    return net

def ResNet_v2(net, training=False):
    net = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(5,5),stride=(2,2),padding='SAME',activation_fn=tf.nn.relu, scope='Conv1') # size = (75, 75)
    net = proposed_residual(net, name='block1', block_num=2, size='SAME', is_training=training)   # size = (75, 75)
    net = proposed_residual(net, name='block2',block_num=2, size='SAME', is_training=training)
    net = proposed_residual(net, name='block3',block_num=2, size='SAME', is_training=training)

    net = proposed_residual(net, name='block4', block_num=2, size='VALID', is_training=training)    # size = (38, 38)
    net = proposed_residual(net, name='block5', block_num=2, size='SAME', is_training=training)
    net = proposed_residual(net, name='block6', block_num=2, size='SAME', is_training=training)
    net = proposed_residual(net, name='block7', block_num=2, size='SAME', is_training=training)

    net = proposed_residual(net, name='block8', block_num=2, size='VALID', is_training=training)    # size = (19, 19)
    net = proposed_residual(net, name='block9', block_num=2, size='SAME', is_training=training)
    net = proposed_residual(net, name='block10', block_num=2, size='SAME', is_training=training)
    net = proposed_residual(net, name='block11', block_num=2, size='SAME', is_training=training)
    net = proposed_residual(net, name='block12', block_num=2, size='SAME', is_training=training)
    net = proposed_residual(net, name='block13', block_num=2, size='SAME', is_training=training)

    net = proposed_residual(net, name='block14', block_num=2, size='VALID', is_training=training)    # size = (10, 10)
    net = proposed_residual(net, name='block15', block_num=2, size='SAME', is_training=training)
    net = proposed_residual(net, name='block16', block_num=2, size='SAME', is_training=training)

    net = tf.contrib.layers.avg_pool2d(net, kernel_size=(10,10), stride=(2,2),padding='VALID',scope='AVG')

    net = tf.contrib.layers.flatten(net, scope='flatten')

    net = tf.contrib.layers.fully_connected(net,num_outputs=60,activation_fn=None, scope='Layer')
    
    return net
