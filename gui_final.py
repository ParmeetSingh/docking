#!/usr/bin/env python


#############################################################################
##
## Copyright (C) 2013 Riverbank Computing Limited.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:BSD$
## You may use this file under the terms of the BSD license as follows:
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##   * Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##   * Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
##     the names of its contributors may be used to endorse or promote
##     products derived from this software without specific prior written
##     permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
## "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
## LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
## A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
## OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
## SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
## LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
## THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
## $QT_END_LICENSE$
##
#############################################################################

import os
from urllib.request import urlopen
from subprocess import Popen, PIPE, STDOUT
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)
import subprocess
import sys
import time
from PyQt5.QtWidgets import QPushButton, QMainWindow, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5 import QtWidgets,QtCore, QtGui


class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self,widgets):
        super(Worker, self).__init__()
        print("new worker")
        self.working = True
        self.widgets = widgets

    def work(self):
        #while self.working:
        string = "python3 docking-identification_yolo_weights_inputv2.py"
        string = string + " -w " + self.widgets.model_link.text()
        string = string + " -cfg " + self.widgets.model_cfg_link.text()
        
        if self.widgets.radioButton2.isChecked() == True:
                video_link = self.widgets.file_link.text()
                string = string + " -src video -video "+video_link
        hoop_size = ""
        if self.widgets.bigRadioButton.isChecked()==True:
                hoop_size = "big"
        elif self.widgets.smallRadioButton.isChecked()==True:
                hoop_size = "small"
        else:
                hoop_size = self.widgets.custom_diameter.text()                                        
        string = string + " -s "+hoop_size
        medium = 'air'
        if self.widgets.waterWidgetsCheckBox.isChecked()==True:
                medium = 'water'
        if self.widgets.save_image_data.isChecked()==True:
                string = string + " -save_train True"
        if self.widgets.save_video.isChecked()==True:
                string = string + " -save_video True"
        if self.widgets.showLightPositions.isChecked()==True:
                string = string + " -l True"                                    
        string = string + " -m "+ medium
        string = string + " -skip " + str(self.widgets.spinBox.value())
        string = string + " -p1 " + str(self.widgets.m1_port.text())
        string = string + " -p2 " + str(self.widgets.m2_port.text())
        print(string)                                 
        self.p1 = subprocess.Popen(string.split(" "))
            #print("I'm running")
            #time.sleep(1)
        #self.finished.emit()  
        




class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.showLightPositions = QCheckBox("Show light position")
        self.save_image_data = QCheckBox("Save data")
        self.save_video = QCheckBox("Save Video")
        self.showLightPositions.setChecked(True)

        self.waterWidgetsCheckBox = QCheckBox("water")

        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createBottomLeftTabWidget()
        self.createBottomRightGroupBox()
        #self.createProgressBar()

        styleComboBox.activated[str].connect(self.changeStyle)
        #self.useStylePaletteCheckBox.toggled.connect(self.changePalette)
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.waterWidgetsCheckBox)
        topLayout.addWidget(self.showLightPositions)
        topLayout.addWidget(self.save_image_data)
        topLayout.addWidget(self.save_video)
        

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)
        #mainLayout.addWidget(self.progressBar, 3, 0, 1, 2)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.setWindowTitle("Docking System Configuration")
        self.changeStyle('Windows')

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        #if (self.useStylePaletteCheckBox.isChecked()):
        QApplication.setPalette(QApplication.style().standardPalette())
        #else:
         #   QApplication.setPalette(self.originalPalette)

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) / 100)
    def getfiles(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', "." , 'Videos (*.mp4 *.jpg *.avi *.MP4)')
        self.file_link.setText(fileName)
    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Source/Ports")

        radioButton1 = QRadioButton("Stream")
        openVideo = QPushButton("Open Video")
        self.file_link = QLineEdit('stream3.mp4')
        #lineEdit = QLineEdit('rtsp://admin:admin@192.168.214.40/h264.sdp?res=half&x0=0&y0=0&x1=1920&y1=1080&qp=16&doublescan=0&ssn=41645')
       
        radioButton1.toggled.connect(openVideo.setDisabled)
        radioButton1.toggled.connect(self.file_link.setDisabled)
        self.radioButton2 = QRadioButton("Video")
        #radioButton2.toggled.connect(lineEdit.setDisabled)
        
        #radioButton3 = QRadioButton("Radio button 3")
        radioButton1.setChecked(True)
        
        
        openVideo.clicked.connect(self.getfiles)
        self.label = QLabel("Ports:")
        self.m1_port = QLineEdit('51110')
        self.m2_port = QLineEdit('51120')
        #self.tst_btn = QPushButton("Test Connection")
        

        
        layout = QGridLayout()
        layout.addWidget(radioButton1, 0, 0, 1, 2)
        #layout.addWidget(lineEdit, 1, 0, 1, 2)
        layout.addWidget(self.radioButton2, 1, 0, 1, 2)
        layout.addWidget(self.file_link, 2, 0)
        layout.addWidget(openVideo, 2, 1, 1, 1)
        
        layout.addWidget(self.label, 3, 0)
        layout.addWidget(self.m1_port, 4, 0,1,1)
        layout.addWidget(self.m2_port, 4, 1,1,1)
        #layout.addWidget(self.tst_btn, 4, 0)
        #layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)    

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Start/Stop and Tests")

        self.defaultPushButton = QPushButton("Start System")
        self.defaultPushButton.setDefault(True)
        #defaultPushButton.clicked.connect(self.on_button_clicked)
        
        self.togglePushButton = QPushButton("End System")
        #self.togglePushButton.setCheckable(True)
        self.togglePushButton.setDisabled(True)
        #togglePushButton.setChecked(True)
        
        self.defaultPushButton.clicked.connect(self.start_loop)
        

        #flatPushButton = QPushButton("Flat Push Button")
        #flatPushButton.setFlat(True)
        
        self.tst_btn = QPushButton("Test UDP Connection")
        self.stream_tst_btn = QPushButton("Test Stream Connection")
        
        self.stream_tst_btn.clicked.connect(self.ping_to_stream)
        self.tst_btn.clicked.connect(self.internet_on)
        self.error_dialog = QtWidgets.QErrorMessage()

        layout = QVBoxLayout()
        layout.addWidget(self.defaultPushButton)
        layout.addWidget(self.togglePushButton)
        layout.addWidget(self.tst_btn)
        layout.addWidget(self.stream_tst_btn)
        #layout.addWidget(flatPushButton)
        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)
    def ping_to_stream(self):
        hostname = "192.168.214.40"
        response = os.system("ping -c 1 " + hostname)
        # and then check the response...
        if response == 0:
            pingstatus = "Network Active: Ping successful to 192.168.214.40"
        else:
            pingstatus = "Network Error: Cant connect to 192.168.214.40"

        self.error_dialog.showMessage(pingstatus)
        return pingstatus
    
    def internet_on(self):
        status = ""
        try:
            response = urlopen('https://www.google.com/', timeout=10)
            status = "Internet connection is on"
        except: 
            status = "Internet connection is off"  
        self.error_dialog.showMessage(status)      
    def start_loop(self):
        print("hello")
        self.thread = QThread()
        self.worker = Worker(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)  # begin our worker object's loop
        self.togglePushButton.clicked.connect(self.stop_loop)  # stop the loop on the stop button click
        #self.togglePushButton.clicked.connect(self.worker.finished.emit)
        self.worker.finished.connect(self.loop_finished)  # do something in the gui when the worker loop ends
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)  # have worker mark itself for deletion
        self.thread.finished.connect(self.thread.deleteLater)  # have thread mark itself for deletion
        
        self.defaultPushButton.setDisabled(True)
        self.togglePushButton.setDisabled(False)
        self.thread.start()        
    
    def stop_loop(self):
        self.worker.p1.kill()
        self.worker.finished.emit()
        self.defaultPushButton.setDisabled(False)
        self.togglePushButton.setDisabled(True)

    def loop_finished(self):
        print('Looped Finished')
    
    def createBottomLeftTabWidget(self):
        self.bottomLeftTabWidget = QGroupBox("Hoop Size")
        self.bigRadioButton = QRadioButton("Big (84 cm diameter)")
        self.smallRadioButton = QRadioButton("Small (68 cm diameter)")
        self.customRadioButton = QRadioButton("Custom diameter in cm")
        self.custom_diameter = QLineEdit('70')
        #lineEdit = QLineEdit('rtsp://admin:admin@192.168.214.40/h264.sdp?res=half&x0=0&y0=0&x1=1920&y1=1080&qp=16&doublescan=0&ssn=41645')
       
        self.bigRadioButton.toggled.connect(self.custom_diameter.setDisabled)
        self.smallRadioButton.toggled.connect(self.custom_diameter.setDisabled)
        self.bigRadioButton.setChecked(True)
        
        
        layout = QGridLayout()
        layout.addWidget(self.bigRadioButton, 0, 0, 1, 2)
        layout.addWidget(self.smallRadioButton, 1, 0, 1, 2)
        layout.addWidget(self.customRadioButton, 2, 0, 1, 2)
        layout.addWidget(self.custom_diameter, 3, 0, 1, 2)
        #layout.addStretch(1)
        self.bottomLeftTabWidget.setLayout(layout)

    def getfilesModel(self):
        modelfileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', "." , '*.hd5')
        self.model_link.setText(modelfileName)
    def getfilesModelConfig(self):
        modelCfgfileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', "." , '*.cfg')
        self.model_cfg_link.setText(modelCfgfileName)    
    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("Frame processing")
        self.bottomRightGroupBox.setCheckable(True)
        self.bottomRightGroupBox.setChecked(True)

        
        label = QLabel("Skip frame")
        self.spinBox = QSpinBox(self.bottomRightGroupBox)
        self.spinBox.setValue(10)
        
        model = QPushButton("Upload Model")
        self.model_link = QLineEdit('weights/yolo_mobilenet_weights_v11_v2.hd5')
        model.clicked.connect(self.getfilesModel)
        
        
        model_cfg = QPushButton("Upload Model Config")
        self.model_cfg_link = QLineEdit('cfg/yolo_mobilenet_weights_v11_v2.cfg')
        model_cfg.clicked.connect(self.getfilesModelConfig)

        layout = QGridLayout()
        layout.addWidget(label, 0, 0)
        layout.addWidget(self.spinBox, 0, 1, 1, 2)
        layout.addWidget(self.model_link, 1, 0)
        layout.addWidget(model, 1, 1, 1, 1)
        layout.addWidget(self.model_cfg_link, 3, 0)
        layout.addWidget(model_cfg, 3, 1, 1, 1)
        #layout.addWidget(dateTimeEdit, 2, 0, 1, 2)
        #layout.addWidget(slider, 3, 0)
        #layout.addWidget(scrollBar, 4, 0)
        #layout.addWidget(dial, 3, 1, 2, 1)
        layout.setRowStretch(5, 1)
        self.bottomRightGroupBox.setLayout(layout)

    def createProgressBar(self):
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 10000)
        self.progressBar.setValue(0)

        timer = QTimer(self)
        timer.timeout.connect(self.advanceProgressBar)
        timer.start(1000)


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec_()) 
