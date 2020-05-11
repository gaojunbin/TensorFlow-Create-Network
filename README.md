# TensorFlow搭建网络

+ 环境  
macOS Catalina 10.15.1  
tensorflow 1.15.0  
conda 4.7.12  

+ 文件架构  
>
>- checkpoint --> 训练过程中保存权重与偏置的文件夹  
>- logs --> tensorboard文件
>- 底层搭网络 --> 通过底层tf库搭建几个经典网络
>- Networks_Visualization --> 用于保存可视化网络的文件夹，不同网络在TensorBoard中可视化，便于搭建新网络时测试正确性  
>- text --> 用于存放训练过程中的loss  
>- Train --> 用于存放训练集与测试集的文件夹，主要由data_reload.py操作  
>- Test --> 用于手动验证测试训练的结果，主要有TestSinglefile.py操作
>- data_reload.py --> 训练集与测试集加载文件  用于读取图片与对应标签生成可被训练的列表  
>- Network.py --> 训练时所用的网络
>- train.py --> 训练  
>- TestSinglefile.py --> 手动测试训练结果，打印输出
>

