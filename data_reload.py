# -*- coding: utf-8 -*-
"""
Created on 2019.12.4
@author: junbin
数据集加载
"""

import cv2
import numpy as np
import os
import random


def Sequential_disruption(train_data, train_label, test_data, test_label):
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
    return distrupted_train_data, distrupted_train_label, distrupted_test_data, distrupted_test_label


def read_train_data():
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    time = 0
    for i in range(1, 3):
        ls_dir = "./Train/" + str(i) + "/"
        dirs = os.listdir(ls_dir)
        num = len(dirs)
        train_num = int(0.95 * num)
        su = []
        for i2 in range(0, num):
            su.append(i2)
        num_train = sorted(random.sample(su, train_num))
        for i3 in range(0, num):
            img = cv2.imread(ls_dir + dirs[i3])
            if img is None or img.shape[0] != 224 or img.shape[1] != 224:
                print("图片不存在或图片尺寸不正确，已跳过")
                continue
            else:
                print(img.shape[0])
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
        print("time=%d"%time)
    train_data, train_label, val_data, val_label = Sequential_disruption(
        train_data, train_label, val_data, val_label)
    return train_data, train_label, val_data, val_label
    # return np.array(train_data,dtype=np.float32),np.array(train_label,dtype=np.float32),np.array(val_data,dtype=np.float32),np.array(val_label,dtype=np.float32)


def read_test_data():
    test_data = []
    test_label = []
    time = 0
    for i in range(1, 11):
        ls_dir = "./Test/" + str(i) + "/" + str(j) + "/"
        dirs = os.listdir(ls_dir)
        num = len(dirs)
        for i3 in range(0, num):
            img = cv2.imread(ls_dir + dirs[i3])
            img = img/255.
            if img is None or img.shape[0] != 224 or img.shape[1] != 224:
                print("图片不存在或图片尺寸不正确，已跳过")
                continue
            else:
                test_data.append(img)
                label = np.zeros((2))
                label[time] = 1.0
                label = label.tolist()
                test_label.append(label)
        time = time+1
        print(time)
    return test_data, test_label
    # return np.array(test_data,dtype=np.float32),np.array(test_label,dtype=np.float32)


def main():
    read_train_data()


if __name__ == '__main__':
    main()
