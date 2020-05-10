import tensorflow as tf
from Network import *
from Data_reload import *

class Train:
    def __init__(self):
        self.net=VGG16()
        self.forward=self.net.vgg16     #选择需要的网络
        self.datasets = None
        self.label = None
        self.test_data = None
        self.test_label = None
        self.ckpt_dir = './checkpoint'
        self.input_data = tf.compat.v1.placeholder(tf.float32, [None,224,224,3], name = "input_data")#定义输入
        self.supervised_label = tf.compat.v1.placeholder(tf.float32, [None, 2], name = "label")#定义标签
        self.BATCH_SIZE=5
        self.STEPS = 10000000           #最大步数
        self.LEARNING_RATE_BASE = 0.00001  # 最初学习率
        self.LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
        self.LEARNING_RATE_STEP = 1000  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/mini_batch
        self.global_step =tf.compat.v1.train.get_or_create_global_step() #步数
        self.learning_rate = tf.compat.v1.train.exponential_decay(self.LEARNING_RATE_BASE, self.global_step, self.LEARNING_RATE_STEP, self.LEARNING_RATE_DECAY, staircase=True)#学习率衰减
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
    def save_variable_list(self):
        '''保存多余变量'''
        var_list = tf.compat.v1.trainable_variables()
        if self.global_step is not None:
            var_list.append(self.global_step)
        g_list = tf.compat.v1.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        return var_list
    def backward(self):
        y = self.forward(self.input_data,self.is_training)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.supervised_label, logits=y))#二次代价函数:预测值与真实值的误差
            tf.summary.scalar('loss', cross_entropy) #生成loss标量图
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.supervised_label, 1))  #结果存放在一个布尔型列表中tf.argmax(prediction,1)返回的是对于任一输入x预测到的标签值，tf.argmax(y_,1)代表正确的标签值
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#求准确率
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)#梯度下降法:数据太庞大,选用AdamOptimizer优化器
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            with tf.name_scope('init'):
                init_op = tf.global_variables_initializer()
            sess.run(init_op)
            var_list = self.save_variable_list()
            saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=5)
            #读取是否需要加载之前的模型文件
            if tf.train.latest_checkpoint(self.ckpt_dir) is not None:
                saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_dir))
            writer = tf.summary.FileWriter("./logs", sess.graph)
            # 训练模型。
            for i in range(self.STEPS):
                start = (i * self.BATCH_SIZE) % int(len(self.datasets)/self.BATCH_SIZE)*self.BATCH_SIZE
                end = start + self.BATCH_SIZE
                train_data = self.datasets[start:end]
                train_label = self.label[start:end]
                print("start:",start,"end:",end,"batchsize:",self.BATCH_SIZE,"datasetslen:",len(self.datasets))
                sess.run(train_step, feed_dict={self.input_data: train_data, self.supervised_label: train_label,self.is_training:True})
                #writer.add_summary(summary_str, i)
                if i % 100 == 0:
                    train_loss = sess.run(cross_entropy, feed_dict={self.input_data: train_data, self.supervised_label: train_label,self.is_training:False})
                    loss = sess.run(cross_entropy, feed_dict={self.input_data: self.test_data, self.supervised_label: self.test_label,self.is_training:False})
                    train_accuracy = accuracy.eval(feed_dict={self.input_data: train_data, self.supervised_label: train_label,self.is_training:False})
                    test_accuracy = accuracy.eval(feed_dict={self.input_data: self.test_data, self.supervised_label: self.test_label,self.is_training:False})
                    print("After %d step(s), train accuracy is %g, val accuracy is %g, train loss is %g, val loss is %g" % (i, train_accuracy,test_accuracy ,train_loss , loss))
                if test_accuracy>0.8 or i>4000:
                    saver.save(sess, './checkpoint/variable', global_step=i)
                    print("精度已达要求或训练次数已达上限，参数已保存，训练终止")
                    break
def mkdir(name):
    '''创建文件夹'''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

if __name__ == '__main__':
    mkdir('./checkpoint')
    dt = Datasets()
    main = Train()
    main.datasets, main.label, main.test_data, main.test_label = dt.read_train_data()
    main.backward()
