import tensorflow as tf
from Network import *
from Data_reload import *

class Test:
    def __init__(self):
        self.net=VGG16()
        self.forward=self.net.vgg16     #选择需要的网络
        self.ckpt_dir = './checkpoint'
        self.testdata = None
        self.input_data = tf.compat.v1.placeholder(tf.float32, [None,224,224,3], name = "input_data")#定义输入
        self.LEARNING_RATE_STEP = 1000  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/mini_batch
        self.global_step =tf.compat.v1.train.get_or_create_global_step() #步数
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
    def test(self):
        y = self.forward(self.input_data,False)
        
        with tf.Session() as sess:
            with tf.name_scope('init'):
                init_op = tf.global_variables_initializer()
            sess.run(init_op)
            var_list = self.save_variable_list()
            saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=5)
            #读取是否需要加载之前的模型文件
            if tf.train.latest_checkpoint(self.ckpt_dir) is not None:
                saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_dir))
                result_calculator = sess.run(y,feed_dict={self.input_data:self.testdata})
                result_class = sess.run(tf.argmax(y, 1),feed_dict={self.input_data:self.testdata})[0]+1
                print(result_calculator)
                print("所属类别：",result_class)
            else:
                print("未找到可用模型")
            

if __name__ == '__main__':
    dt = Datasets()
    main = Test()
    main.testdata = dt.read_test_file('小狗_2.jpg')
    main.test()
