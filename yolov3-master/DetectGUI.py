from PyQt5 import QtCore, QtGui, QtWidgets
class Ui_TabWidget(object):
    def setupUi(self, TabWidget):
        TabWidget.setObjectName("TabWidget")    #设置创建的是"TabWidget"
        #TabWidget.resize(789, 619)
        TabWidget.resize(1200, 1000)             # 设置主窗口的宽和高

        ############################################################################
        # 第一个子窗口的信息
        '''
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("视频检测和识别")           #"第一个子窗口"
        # 创建的Button的位置信息

        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(10, 50, 75, 50))     #Button2的位置信息
        self.pushButton.setObjectName("pushButton_2")               #设置创建的是"pushButton2"

        # 设置显示的区域
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(110, 25, 640, 480))
        self.label.setText("")
        self.label.setObjectName("label")

        TabWidget.addTab(self.tab, "")  # 第一个子窗口添加成功
        '''
        ############################################################################
        self.tab1 = QtWidgets.QWidget()  # "第二个子窗口"
        self.tab1.setObjectName("图像检测和识别")
        TabWidget.addTab(self.tab1, "")

        self.pushButton1 = QtWidgets.QPushButton(self.tab1)
        self.pushButton1.setGeometry(QtCore.QRect(10, 50, 100, 50))       #Button1的位置信息
        self.pushButton1.setObjectName("pushButton")                    #设置创建的是"pushButton"
        self.pushButton12 = QtWidgets.QPushButton(self.tab1)
        self.pushButton12.setGeometry(QtCore.QRect(10, 150, 100, 50))  # Button12的位置信息
        self.pushButton12.setObjectName("pushButton12")  # 设置创建的是"pushButton"

        self.label1 = QtWidgets.QLabel(self.tab1)
        self.label1.setGeometry(QtCore.QRect(110, 25, 640, 480))
        self.label1.setText("")
        self.label1.setObjectName("label2")
        TabWidget.addTab(self.tab1, "")          # 第一个子窗口添加成功

        # 其他设置
        self.retranslateUi(TabWidget)
        TabWidget.setCurrentIndex(0)
        #self.pushButton.clicked.connect(TabWidget.videoprocessing)          #将按键1与事件相连
        self.pushButton12.clicked.connect(TabWidget.imageprocessing2)  # 将按键1与事件相连
        self.pushButton1.clicked.connect(TabWidget.imageprocessing)        #将按键2与事件相连
        QtCore.QMetaObject.connectSlotsByName(TabWidget)

    def retranslateUi(self, TabWidget):
        _translate = QtCore.QCoreApplication.translate
        TabWidget.setWindowTitle(_translate("TabWidget", "安检图像违禁品检测与识别软件"))       # 设置主窗口的名字
        #self.pushButton.setText(_translate("TabWidget", "打开视频"))            #设置按键的名字
        self.pushButton12.setText(_translate("TabWidget", "开始检测"))  # 设置按键的名字
        self.pushButton1.setText(_translate("TabWidget", "打开图像"))          #设置按键的名字
        # 设置三个分页的名字
        #TabWidget.setTabText(TabWidget.indexOf(self.tab), _translate("TabWidget", "视频检测和识别"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab1), _translate("TabWidget", "违禁品检测与识别"))
