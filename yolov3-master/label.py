'''
标准文件格式转换
输入：XML标注文件的位置
'''

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
 
sets = ['train', 'test','val']
 
#classes = ["liquid","knife","lighter1","lighter2","scissors","power_bank"]
classes = ["handgun","lighter1","lighter2", "power_bank", "knife", "liquid", "scissors", "laptop", "1", '2', '3', '4', '5', '6', '7', '8']

"""
在比赛数据集中:
铁壳打火机: 1
黑钉打火机: 2
刀具:3
电池:4
剪刀:5
液体:6
笔记本:7
手枪:8
"""
classes_dict={'8':"handgun", '1':"lighter1", '2':"lighter2", '4': "power_bank",
               '3': "knife", '6':"liquid", '5':"scissors", '7':"laptop", 'latop':'laptop', 'latpop':'laptop', 'laｐtop':'laptop'}
 
def convert(size, box):
    dw = 1./size[0]           # 归一化的时候就是使用宽度除以整个image_size的宽度 
    dh = 1./size[1]           # 归一化的时候就是使用高度除以整个image_size的高度
    x = (box[0] + box[1])/2.0 # 使用(xmin+xmax)/2得到x的中心点
    y = (box[2] + box[3])/2.0 # 使用(ymin+ymax)/2得到y的中心点
    w = box[1] - box[0]       # 然后宽度就是使用xmax-xmin计算得到
    h = box[3] - box[2]       # 然后高度就是使用ymax-ymin计算得到
    x = x*dw                  # 归一化中心坐标x
    w = w*dw                  # 归一化bbox宽度
    y = y*dh                  # 归一化中心坐标y
    h = h*dh                  # 归一化bbox高度
    return (x,y,w,h)

 
 
def convert_annotation(image_id):
    image_id = image_id.replace('.jpg', '').replace('data/images/', '')
    
    try:
        in_file = open('data/Annotations/%s.xml' % (image_id))
        out_file = open('data/labels/%s.txt' % (image_id), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    except:
        out_file = open('data/labels/%s.txt' % (image_id), 'w')
        #print("No xml file: ", image_id)
        return 
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        if cls in classes_dict.keys():
        	cls = classes_dict[cls]

        #if cls not in classes or int(difficult) == 1:
        #    continue
        
        try:	
            cls_id = classes.index(cls)
        except:
            print(cls)
            print(image_id)
            continue
        xmlbox = obj.find('bndbox')
        # 获取标注中bbox的数据并以元组方式返回
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        # 向convert传入size = (w,h),和b,注释中分别是(xmin,xmax,ymin,ymax)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
 
 
wd = getcwd()
print(wd)
for image_set in sets:
    if not os.path.exists('data/labels/'):
        os.makedirs('data/labels/')
    image_ids = open('data/%s.txt' % (image_set)).read().strip().split()
    list_file = open('data/%s.txt' % (image_set+'labels'), 'w')
    for image_id in image_ids:
        image_id = image_id.replace('.jpg', '').replace('data/images/', '')
        list_file.write('data/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
