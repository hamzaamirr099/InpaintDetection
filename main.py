# To build it -> pyinstaller.exe --onefile --windowed --icon=logo.ico main.py

import os
from pickle import TRUE
from PIL import Image
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtCore 
from PyQt5.QtGui import *
from os import path
import sys
import random
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMessageBox
from numpy import imag
from torch import t
import predict


class gpApp(QMainWindow):

    imagePath = "" # Path of the image that will be predicted
    
    def __init__(self):
        
        super(gpApp , self).__init__()
        loadUi("gpApp.ui", self)
        self.Buttons()
        self.checkButton.setEnabled(False)

    def showmessagebox(self, title, message):

        msgbox = QMessageBox()
        msgbox.resize(300,200)
        msgbox.setIcon(QMessageBox.Information)
        msgbox.setWindowTitle(title)
        msgbox.setText(message)
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()

    def Buttons(self):
        self.browseButton.clicked.connect(self.browse)
        self.checkButton.clicked.connect(self.check)
        self.exitButton.clicked.connect(self.exit)

    def browse(self):
        self.lineEditResult.setText("Result")
        self.lineEditResult.setStyleSheet("color: grey; background-color: rgba(0, 0, 0, 100);")
        fName = QFileDialog.getOpenFileName(self, "Select the image", "../", "*.png *.xpm *.jpg")
        self.imagePath = fName[0]
        # print(imagePath)
        if(self.imagePath):
            self.checkButton.setEnabled(True)

        pixmap_image = QtGui.QPixmap(fName[0]).scaled(self.imageLabel.width(), self.imageLabel.height(), QtCore.Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap_image)


    def check(self):
        
        print(self.imagePath)
        result = predict.predictImage(self.imagePath)
        print(result)
        if(result > 0.5):
            self.lineEditResult.setStyleSheet("color: red; background-color: rgba(0, 0, 0, 100);")
            self.lineEditResult.setText("Inpainted")
        else:
            self.lineEditResult.setStyleSheet("color: green; background-color: rgba(0, 0, 0, 100);")
            self.lineEditResult.setText("Original")

    def exit(self):
        self.close()

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    # widget = QtWidgets.QStackedWidget()
    window = gpApp()

    window.setWindowTitle("Inpaint Detection")
    window.setWindowIcon(QtGui.QIcon('logo.ico'))
    
    window.show()
    app.exec_()