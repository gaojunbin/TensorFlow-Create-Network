import tensorflow as tf

class VGG16:
    def __init__(self):
        self.keep_prob=0.8
    def vgg16(self,net,is_training=True):
        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN1')
        net = tf.contrib.layers.conv2d(net,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv1')
        #net = tf.contrib.layers.conv2d(net,num_outputs=4,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv2')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool1')
        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN2')



        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN2')
        net = tf.contrib.layers.conv2d(net,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv2')
        #net = tf.contrib.layers.conv2d(net,num_outputs=8,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv4')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool2')
        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN3')


        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN3')
        net = tf.contrib.layers.conv2d(net,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv3')
        #net = tf.contrib.layers.conv2d(net,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv1')
        #net = tf.contrib.layers.conv2d(net,num_outputs=16,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv1')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool3')
        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN4')


        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN4')
        net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv4')
        #net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv5')
        #net = tf.contrib.layers.conv2d(net,num_outputs=32,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv1')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool4')
        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN5')


        #net = tf.contrib.layers.batch_norm(net, is_training=is_training, scope='BN5')
        net = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv5')
        #net = tf.contrib.layers.conv2d(net,num_outputs=64,kernel_size=(3,3),stride=(1,1),padding='valid',activation_fn=tf.nn.relu, scope='Conv1')
        net = tf.contrib.layers.max_pool2d(net,kernel_size=(2,2),stride=(2,2),padding='valid', scope='Maxpool5')
        net = tf.contrib.layers.flatten(net, scope='flatten')
        ## funcl layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=256,activation_fn=tf.nn.relu, scope='fully_connected1')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout1')
        ## func2 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=128,activation_fn=tf.nn.relu, scope='fully_connected2')
        net = tf.contrib.layers.dropout(net,keep_prob=self.keep_prob,is_training=is_training, scope='dropout2')
        ## func3 layer ##
        net = tf.contrib.layers.fully_connected(net,num_outputs=2,activation_fn=None, scope='logits')

        return net
