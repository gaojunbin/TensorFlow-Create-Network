# -*- coding: utf-8 -*-
"""
Created on 2019.12.4
latest modify 2020.5.10
@author: Junbin
@note: Dataset reload
"""

import cv2
import numpy as np
import os
import random

class Datasets():
    def Sequential_disruption(self,train_data, train_label, test_data, test_label):
        # 此函数对训练集与数据集进行随机的打乱，防止其整齐
        train_num = len(train_label)
        test_num = len(test_label)
        train_seq_distruption = [i for i in range(0, train_num)]
        test_seq_distruption = [i for i in range(0, test_num)]
        random.shuffle(train_seq_distruption)
        random.shuffle(test_seq_distruption)
        distrupted_train_data = []
        distrupted_train_label = []
        distrupted_test_data = []
        distrupted_test_label = []
        for i in train_seq_distruption:
            distrupted_train_data.append(train_data[i])
            distrupted_train_label.append(train_label[i])
        for i in test_seq_distruption:
            distrupted_test_data.append(test_data[i])
            distrupted_test_label.append(test_label[i])
        return np.array(distrupted_train_data), np.array(distrupted_train_label), np.array(distrupted_test_data), np.array(distrupted_test_label)

    def read_train_data(self):
        train_data = []
        train_label = []
        val_data = []
        val_label = []
        time = 0
        for i in range(1, 3):
            ls_dir = "./Train/" + str(i) + "/"
            dirs = os.listdir(ls_dir)
            num = len(dirs)
            train_num = int(0.7 * num)
            val_num = num - train_num
            su = []
            for i2 in range(0, num):
                su.append(i2)
            num_train = sorted(random.sample(su, train_num))
            for i3 in range(0, num):
                img = cv2.imread(ls_dir + dirs[i3])
                if img is None :
                    print("%s图片不存在，已跳过"%(ls_dir + dirs[i3]))
                    continue
                elif img.shape[0] != 224 or img.shape[1] != 224:
                    print("图片尺寸不正确，(%d,%d)，已跳过"%(img.shape[0],img.shape[1]))
                    continue
                else:
                    img = img / 255.
                    if (int(i3) in num_train):
                        train_data.append(img)
                        label = np.zeros(2)
                        label[time] = 1.0
                        label = label.tolist()
                        train_label.append(label)
                    else:
                        val_data.append(img)
                        label = np.zeros((2))
                        label[time] = 1.0
                        label = label.tolist()
                        val_label.append(label)
            time = time+1
            print("class=%d,train_num=%d,val_num=%d"%(time,train_num,val_num))
        print("训练集读取完毕,共%d个分类"%time)
        # print("训练集形状",(np.array(train_data)).shape,"标签形状：",(np.array(train_label)).shape)
        train_data, train_label, val_data, val_label = self.Sequential_disruption(train_data, train_label, val_data, val_label)
        return train_data, train_label, val_data, val_label

    def read_test_file(self,filename):
        test_data = []
        test_label = []
        objectfile = "./Test/"+str(filename)
        img = cv2.imread(objectfile)  
        if img is None :
            print("%s图片不存在，已跳过"%(ls_dir + dirs[i3]))
            return None
        elif img.shape[0] != 224 or img.shape[1] != 224:
            print("图片尺寸不正确，(%d,%d)，已跳过"%(img.shape[0],img.shape[1]))
            return None
        else:
            img = img / 255.
            test_data.append(img)
            return np.array(test_data)


def main():
    datasets = Datasets()
    datasets.read_train_data()


if __name__ == '__main__':
    main()
