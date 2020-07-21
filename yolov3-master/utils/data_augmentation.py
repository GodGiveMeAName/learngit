# -*- coding:utf-8 -*-

'''
数据增广：
    旋转
    亮度、饱和度、对比度
    色彩抖动
'''

import os
import cv2
import xml.dom.minidom
from xml.dom.minidom import Document  
import math
import codecs
import random 
import xml.etree.ElementTree as ET
import numpy as np

classes = ['handgun','lighter1','lighter2','power_bank',
            'knife','liquid','scissors','laptop']
class_dict = {
'1':'lighter1',
'2':'lighter2',
'3':'knife',
'4':'power_bank',
'5':'scissors',
'6':'liquid',
'7':'laptop',
'8':'handgun'
}


# 获取路径下所有文件的完整路径，用于读取文件用
def GetFileFromThisRootDir(dir,ext = None):
    allfiles = []
    needExtFilter = (ext != None)
    for root,dirs,files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles

# 图像旋转
def im_rotate(im, angle, center = None, scale = 1.0):
    h, w = im.shape[:2]
    if center is None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    im_rot = cv2.warpAffine(im, M, (w,h))  #应用一个仿射变换
    return im_rot

# 读取xml文件
def readXml(image_id):
    try:
        in_file = open(image_id)
    except:
        return None
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    channels = int(size.find('depth').text)
    bboxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        if cls in class_dict.keys():
            cls = class_dict[cls]
        #cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        # 获取标注中bbox的数据并以list方式返回
        b = [float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text), cls]
        bboxes.append(b)
    return bboxes, w, h, channels

# 写入xml文件,参数中tmp表示路径，imgname是文件名（没有尾缀）ps有尾缀也无所谓
def writeXml(tmp,imgname,w,h,d,bboxes):
    doc = Document()
    annotation = doc.createElement('annotation')  
    doc.appendChild(annotation) 
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    
    folder_txt = doc.createTextNode("VOC2007")
    
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)  
    filename_txt = doc.createTextNode(imgname)  
    filename.appendChild(filename_txt)
    source = doc.createElement('source')  
    annotation.appendChild(source) 
    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("My Database")
    database.appendChild(database_txt)
    
    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("VOC2007")
    annotation_new.appendChild(annotation_new_txt)
    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    #owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)
    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)
    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)
    #onee# 
    #twos# 
    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width') 
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))  
    width.appendChild(width_txt)     
    height = doc.createElement('height') 
    size.appendChild(height)    
    height_txt = doc.createTextNode(str(h))  
    height.appendChild(height_txt)    
    depth = doc.createElement('depth')  
    size.appendChild(depth)   
    depth_txt = doc.createTextNode(str(d))   
    depth.appendChild(depth_txt)  
    #twoe#    
    segmented = doc.createElement('segmented') 
    annotation.appendChild(segmented)    
    segmented_txt = doc.createTextNode("0")  
    segmented.appendChild(segmented_txt)
    for bbox in bboxes:    
        #threes#          
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)           
        name = doc.createElement('name')        
        object_new.appendChild(name)     
        name_txt = doc.createTextNode(str(bbox[4]))       
        name.appendChild(name_txt)           
        pose = doc.createElement('pose')     
        object_new.appendChild(pose)         
        pose_txt = doc.createTextNode("Unspecified")      
        pose.appendChild(pose_txt)         
        truncated = doc.createElement('truncated')      
        object_new.appendChild(truncated)         
        truncated_txt = doc.createTextNode("0")    
        truncated.appendChild(truncated_txt)       
        difficult = doc.createElement('difficult')   
        object_new.appendChild(difficult)        
        difficult_txt = doc.createTextNode("0")      
        difficult.appendChild(difficult_txt)       
        #threes-1#         
        bndbox = doc.createElement('bndbox')      
        object_new.appendChild(bndbox)          
        xmin = doc.createElement('xmin')        
        bndbox.appendChild(xmin)        
        xmin_txt = doc.createTextNode(str(bbox[0]))   
        xmin.appendChild(xmin_txt)      
        ymin = doc.createElement('ymin')   
        bndbox.appendChild(ymin)         
        ymin_txt = doc.createTextNode(str(bbox[1])) 
        ymin.appendChild(ymin_txt)          
        xmax = doc.createElement('xmax')        
        bndbox.appendChild(xmax)         
        xmax_txt = doc.createTextNode(str(bbox[2])) 
        xmax.appendChild(xmax_txt)              
        ymax = doc.createElement('ymax')       
        bndbox.appendChild(ymax)       
        ymax_txt = doc.createTextNode(str(bbox[3]))    
        ymax.appendChild(ymax_txt)   
        
    tempfile = tmp +"%s.xml"%imgname  
    with codecs.open(tempfile, 'w','utf-8') as f:
        doc.writexml(f,indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')
    return 187

def convert(img):
    (rows, cols, ch) = img.shape
    #cv2.imwrite('E:\\yolov3-master\\raw.jpg', img)
    dst = np.array(img)
    #dst = np.zeros((img.shape))
    a = np.random.randn()*15
    tmp = np.random.randn(img.shape[0],img.shape[1],3)*a
    #tmp = np.ones((img.shape))*a
    dst = dst + tmp
    #cv2.imwrite('E:\\yolov3-master\\dst.jpg', dst)
    return dst

def generate_data_by_rotate_gauss(img_dir, anno_dir, output):
    imgs_path=GetFileFromThisRootDir(img_dir)
    if not os.path.isdir(output):
        os.makedirs(output)
    #旋转角的大小，整数表示逆时针旋转
    angles = [i for i in range(360)]#角度im_rotate用到的是角度制
    angle_rad = [angle*math.pi/180.0 for angle in angles] #cos三角函数里要用到弧度制的 
    j=0 # 计数
    angle_num = len(angles)
    for img_path in imgs_path :    
        #读取原图像    
        im = cv2.imread(img_path)
        
        i = random.randint(0,500) # 以百分之五十的几率随机角度旋转图像
        if i > 0 and i < 360:  
            gt_new = []        
            im_rot = im_rotate(im,angles[i])  
            file_name = img_path.split('/')[-1][:-4]    
            #画出旋转后图像
            im_rot = convert(im_rot) # 添加高斯噪声
            cv2.imwrite(os.path.join(output,'_%s_.jpg'%(file_name[29:])),im_rot)
            anno = os.path.join(anno_dir,'%s.xml'%file_name[29:])    
            print(anno)
            #读取anno标签数据
            try:
                [gts,w,h,d] =readXml(anno)
            except:
                continue
            #计算旋转后gt框四点的坐标变换       
            [xc,yc] = [float(w)/2,float(h)/2]      
            for gt in gts:           
                #计算左上角点的变换         
                x1 = (gt[0]-xc)*math.cos(angle_rad[i]) - (yc-gt[1])*math.sin(angle_rad[i]) + xc      
                if x1<0:
                    x1=0          
                if x1>w-1 :
                    x1=w-1                   
                y1 = yc - (gt[0]-xc)*math.sin(angle_rad[i]) - (yc-gt[1])*math.cos(angle_rad[i])
                if y1<0:
                    y1=0   
                if y1>h-1: 
                    y1=h-1  
                #计算右上角点的变换     
                x2 = (gt[2]-xc)*math.cos(angle_rad[i]) - (yc-gt[1])*math.sin(angle_rad[i]) + xc   
                if x2<0:
                    x2=0    
                if x2>w-1: 
                    x2=w-1       
                y2 = yc - (gt[2]-xc)*math.sin(angle_rad[i]) - (yc-gt[1])*math.cos(angle_rad[i])  
                if y2<0: 
                    y2=0             
                if y2>h-1:
                    y2=h-1          
                #计算左下角点的变换         
                x3 = (gt[0]-xc)*math.cos(angle_rad[i]) - (yc-gt[3])*math.sin(angle_rad[i]) + xc  
                if x3<0:
                    x3=0    
                if x3>w-1:
                    x3=w-1                    
                y3 = yc - (gt[0]-xc)*math.sin(angle_rad[i]) - (yc-gt[3])*math.cos(angle_rad[i])   
                if y3<0:
                    y3=0                  
                if y3>h-1:
                    y3=h-1          
                #计算右下角点的变换       
                x4 = (gt[2]-xc)*math.cos(angle_rad[i]) - (yc-gt[3])*math.sin(angle_rad[i]) + xc  
                if x4<0:
                    x4=0                  
                if x4>w-1:
                    x4=w-1          
                y4 = yc - (gt[2]-xc)*math.sin(angle_rad[i]) - (yc-gt[3])*math.cos(angle_rad[i])
                if y4<0:
                    y4=0                 
                if y4>h-1 : 
                    y4=h-1        
                xmin = min(x1,x2,x3,x4)     
                xmax = max(x1,x2,x3,x4)     
                ymin = min(y1,y2,y3,y4)      
                ymax = max(y1,y2,y3,y4)      
                #把因为旋转导致的特别小的 长线型的去掉      
                w_new = xmax-xmin+1         
                h_new = ymax-ymin+1        
                ratio1 = float(w_new)/h_new       
                ratio2 = float(h_new)/w_new      
                if(1.0/6.0<ratio1<6 and 1.0/6.0<ratio2<6 and w_new>9 and h_new>9):       
                    classname = str(gt[4])       
                    gt_new.append([xmin,ymin,xmax,ymax,classname]) 
                
                #写出新的xml         
                writeXml(output,'_%s_'%file_name[29:], w, h, d, gt_new) 
        j = j+1    
        if j%100==0:
            print( '----%s----'%j)


if __name__ == '__main___':
    img_dir = 'E:\\yolov3-master\\data\\images'
    anno_dir = 'E:\\yolov3-master\\data\\Annotations'
    output = 'E:\\yolov3-master\\data\\data_augment\\'
    rotate_main(img_dir, anno_dir, output)
