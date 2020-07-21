#encoding:utf-8
import sys, cv2, time
import os.path as osp
import argparse
from DetectGUI import Ui_TabWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QTabWidget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel,QWidget


# from predict import predictImage

import torch
from detect import Detector

class mywindow(QTabWidget,Ui_TabWidget): #这个窗口继承了用QtDesignner 绘制的窗口
    def __init__(self, opt):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.opt = opt
        self.detector = Detector(cfg=opt.cfg, data_cfg=opt.data_cfg, weights=opt.weights, 
                            img_size=opt.img_size, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    def videoprocessing(self):
        global videoName #在这里设置全局变量以便在线程中使用
        videoName, videoType = QFileDialog.getOpenFileName(self, "打开视频",
                               "",
                               " *.MOV;;*.mp4;;*.avi;;All Files (*)")
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def imageprocessing(self):
        self.imgName, imgType= QFileDialog.getOpenFileName(self,
                                    "打开图片",
                                    "",
                                    #" *.jpg;;*.png;;*.jpeg;;*.bmp")
                                    " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        #利用qlabel显示图片
        # print(str(self.imgName))
        png = QtGui.QPixmap(self.imgName).scaled(self.label1.width(), self.label1.height())#适应设计label时的大小
        self.label1.setPixmap(png)

    def imageprocessing2(self):
        opt = self.opt
        with torch.no_grad():
            start = time.time()
            savepath = self.detector.detect_one_image(self.imgName)
            # savepath = detect_one_image(opt.cfg,
            #                             opt.data_cfg,
            #                             opt.weights,
            #                             image_path=self.imgName,
            #                             output='output',  # output folder
            #                             img_size=416,
            #                             conf_thres=0.5,
            #                             nms_thres=0.5)
            print('cost time is {}s'.format(round(time.time()-start,2))) 
        savepath = osp.abspath(savepath)
        #利用qlabel显示图片
        print(str(savepath))
        png = QtGui.QPixmap(savepath).scaled(self.label1.width(), self.label1.height())#适应设计label时的大小
        self.label1.setPixmap(png)


class Thread(QThread):#采用线程来播放视频
    changePixmap = pyqtSignal(QtGui.QImage)
    def run(self):
        cap = cv2.VideoCapture(videoName)
        print(videoName)
        while (cap.isOpened()==True):
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgbImage = cv2.resize(rgbImage, (640, 480), interpolation=cv2.INTER_CUBIC)

                # 添加对图像帧的处理算法
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)#在这里可以对每帧图像进行处理，
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(0.01) #控制视频播放的速度
            else:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-8cls.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/rbc.data', help='.data file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--image-path', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=608, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.1, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    app = QtWidgets.QApplication(sys.argv)
    window = mywindow(opt)
    window.show()
    sys.exit(app.exec_())








