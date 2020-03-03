# TensorFlow搭建网络

+ 环境  
macOS Catalina 10.15.1  
tensorflow 1.15.0  
conda 4.7.12  

+ 文件架构  
>
> - checkpoint --> 训练过程中保存权重与偏置的文件夹  
> - Networks_Visualization --> 用于保存可视化网络的文件夹，不同网络在TensorBoard中可视化，便于搭建新网络时测试正确性  
>- text --> 用于存放训练过程中的loss  
>- Train --> 用于存放训练集与测试集的文件夹，主要由data_reload.py操作  
>-data_reload.py --> 训练集与测试集加载文件  用于读取图片与对应标签生成可被训练的列表  
>- train.py --> 训练  
>- get_learning_curve.py --> (有待开发及更新）   
>- Inference.py --> (有待开发及更新）  
>（以下是不同的网络，持续更新）  
>- googlenet_v1.py --> GoogLeNet_V1网络  
>- ResNet_v2.py --> ResNet_v2网络  
>- Vgg_16.py --> Vgg_16网络  
>

