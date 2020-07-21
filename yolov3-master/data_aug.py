# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:19:54 2019
数据筛选：
     选则特定数据：处在决策边界上的数据
     输出txt文件
@author: walkingsoul
"""

import argparse
import json
from torch.utils.data import DataLoader
from models import *
from utils.datasets import *
from utils.utils import *
from utils.data_augmentation import *
import torch

hyp = {'giou': 2.666,   # giou loss gain
       'xy': 4.062,     # xy loss gain
       'wh': 4.1845,    # wh loss gain
       'cls': 8.26,     # cls loss gain
       'cls_pw': 1.1,     # cls BCELoss positive_weight
       'obj': 10.61,     # obj loss gain
       'obj_pw': 1.5,     # obj BCELoss positive_weight
       'iou_t': 0.20,   # iou target-anchor training threshold
       'lr0': 0.01,    # initial learning rate
       'lrf': -3.,      # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,        # SGD momentum
       'weight_decay': 0.0005,   # optimizer weight decay L2正则项，不是学习率的衰减
       'class_weights':[2,1.5,2,1.5,1.5,1,1.2,1.5]
       }  

def get_loss(
    cfg,
    data_cfg,
    weights=None,
    img_size=416,
    iou_thres=0.5,
    conf_thres=0.001,
    nms_thres=0.4,
    model=None
):
    ''' 1 模块初始化 '''
    if model is None:
        # Initialize model
        print('初始化模型...')
        device = torch_utils.select_device()
        model = Darknet(cfg, img_size).to(device)
        model.hyp = hyp  # attach hyperparameters to model

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Configure run
    data_cfg = parse_data_cfg(data_cfg)
    nc = int(data_cfg['classes'])  # number of classes
    test_path = data_cfg['train']  # path to test images
    names = load_classes(data_cfg['names'])  # class names
    
    ''' 2 加载图像 '''
    print("***---测试图片路径：---***",test_path)
    dataset = LoadImagesAndLabels(test_path, img_size, 1,rect=False)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=6,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
    
    ''' 3 图像损失检测与保存 '''
    tol_loss = []
    model.eval()
    file_output = open('data_aug_files.txt','w')
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Computing mAP')):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs
        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            cur_loss = compute_loss(train_out, targets, model)[0].item()
            #print('loss:',cur_loss)
            tol_loss.append(cur_loss)
            if cur_loss > 0.02:
                file_output.write(paths[0]+'\n')
        
    print('损失数量', len(tol_loss))
    print('平均损失', sum(tol_loss)/len(tol_loss))
    print('最大损失', max(tol_loss))
    print('最小损失', min(tol_loss))
    file_output.close()



def generate_data(img_file, anno_dir, output):
    
    if not os.path.isdir(output):
        os.makedirs(output)
    #旋转角的大小，整数表示逆时针旋转
    angles = [i for i in range(360)]#角度im_rotate用到的是角度制
    angle_rad = [angle*math.pi/180.0 for angle in angles] #cos三角函数里要用到弧度制的 
    j=0 # 计数
    f = open(img_file,"r")   
    lines = f.readlines()  
    angle_num = len(angles)
    for img_path in lines:    
        #读取原图像 
        img_path.replace('\n','')
        img_path = r'E:/yolov3-master/' + img_path[:-1]
        print(img_path+'___000')
        try:
            im = cv2.imread(img_path)
        except:
        	print(img_path+'___000')
            

        file_name = img_path.split('/')[-1][:-4]    
        
        im_gauss = convert(im) # 添加高斯噪声
        cv2.imwrite(os.path.join(output,'_%s_.jpg'%(file_name)),im_gauss)

        anno = os.path.join(anno_dir,'%s.xml'%file_name)  
        new_anno = os.path.join(output,'_%s_.xml'%file_name)
        #写出新的xml
        try:
            shutil.copyfile(anno, new_anno)
        except:
            pass
        

        j = j+1    
        if j%100==0:
            print( '----%s----'%j)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-8cls.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/rbc.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best_1024_tiny.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--anno-dir', type=str, default='data\\Annotations\\')
    parser.add_argument('--output', type=str, default='data\\data_augment\\')
    parser.add_argument('--img-file', type=str, default='data_aug_files.txt')

    opt = parser.parse_args()
    print(opt)

    #with torch.no_grad():
    #    get_loss(opt.cfg, opt.data_cfg, opt.weights, opt.img_size, opt.iou_thres, opt.conf_thres, opt.nms_thres)

    generate_data(opt.img_file, opt.anno_dir, opt.output)
        
        
        
        
    
