#!/usr/bin/python
# -*- coding: utf-8 -*

'''
将数据集划分成三个部分： 训练集、测试集、验证集
输出对应三个 txt 文件
'''

import os
import random
 

train_percent = 0.8
test_val_percent = 0.2   # 验证集+测试集
val_percent = 0.1        # 验证集

xmlfilepath = 'data/images'         # 图片比标签多 因此选则此路径
total_xml = os.listdir(xmlfilepath)
 
num = len(total_xml)
list = range(num)

train_set_num = int(num * train_percent)
test_val_set_num = int(num * test_val_percent)
val_set_num = int(num * val_percent)

test_val = random.sample(list, test_val_set_num)
val = random.sample(test_val, val_set_num)

# ftrainval = open('data/ImageSets/trainval.txt', 'w')
ftest = open('data/test.txt', 'w')
ftrain = open('data/train.txt', 'w')
fval = open('data/val.txt', 'w')
 
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in test_val:
        #ftrainval.write(name)
        if i in val:
            fval.write(name)
        else:
            ftest.write(name)
    else:
        ftrain.write(name)

# ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
