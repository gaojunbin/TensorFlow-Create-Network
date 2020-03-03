# -*- coding: utf-8 -*-
"""
Created on 2019.12.4
@author: junbin

"""

import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('loss.txt')

step = a[:,0]
train_acc = a[:,1]
val_acc = a[:,2]

plt.plot(step,train_acc,'.-',label='train acc')
plt.plot(step,val_acc,'.-',label='val acc')
plt.legend(loc = 'upper right')
plt.show()