

# Importing needed libraries
# We need sys library to pass arguments into QApplication
import sys
# QtWidgets to work with widgets
from PyQt5 import QtWidgets
# QPixmap to work with images
from PyQt5.QtGui import QPixmap


from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np



# Importing designed GUI in Qt Designer as module
import vt_lpr2

# Importing YOLO v3 module to Detect Objects on image
from vt_lpr_code import yolo3


"""
Start of:
Main class to add functionality of designed GUI
"""

class VideoThread(QThread):
    # def init(self, parent=None):
    #     QThread.init(self, parent)
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_pixmap_signal1 = pyqtSignal(np.ndarray)
    change_pixmap_signal2= pyqtSignal(np.ndarray)
    change_pixmap_signal3 = pyqtSignal(np.ndarray)
    update=pyqtSignal(str)
    update1=pyqtSignal(str)
    update2=pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        # self.pushButton.clicked.connect(self.run)
        # self.thread = MainApp()
        self.abc=MainApp()
        print(MainApp.update_label_object.image_path)



    def run(self):
    #     # capture from web cam

        # new_name='original_image/1.jpg'
        cap = cv2.VideoCapture(MainApp.update_label_object.image_path)
        # out = cv2.VideoWriter("chck1.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        while self._run_flag:
            ret, cv_img = cap.read()
            
            if ret:
                
                frame1,cropped1,cropped2,lab1,lab2,lab3=yolo3(cv_img,"prediction_result.csv")

                self.change_pixmap_signal.emit(cv_img)
                self.change_pixmap_signal1.emit(frame1)
                self.change_pixmap_signal2.emit(cropped1)
                self.change_pixmap_signal3.emit(cropped2)
                self.update.emit(lab1)
                self.update1.emit(lab2)
                self.update2.emit(lab3)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()




class MainApp(QtWidgets.QMainWindow, vt_lpr2.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.update_label_object)

    # Defining function that will be implemented after button is pushed
    # noinspection PyArgumentList
    def update_label_object(self):

        # Showing text while image is loading and processing
        # self.label.setText('Processing ...')

        # Opening dialog window to choose an image file
        # Giving name to the dialog window --> 'Choose Image to Open'
        # Specifying starting directory --> '.'
        # Showing only needed files to choose from --> '*.png *.jpg *.bmp'
        # noinspection PyCallByClass
        MainApp.update_label_object.image_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open',
                                                  '.',
                                                  '*.png *.jpg *.bmp')

        # Variable 'image_path' now is a tuple that consists of two elements
        # First one is a full path to the chosen image file
        # Second one is a string with possible extensions

        # Checkpoint
        # print(type(image_path))  # <class 'tuple'>
        # print(image_path[0])  # /home/my_name/Downloads/example.png
        # print(image_path[1])  # *.png *.jpg *.bmp

        # Slicing only needed full path
        MainApp.update_label_object.image_path = MainApp.update_label_object.image_path[0]
        print(MainApp.update_label_object.image_path)

        
        # create the label that holds the image
        # self.image_label = QLabel(self)
        # self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        # self.textLabel = QLabel('Webcam')
        self.disply_width = 500
        self.display_height = 500
        # self.label = QLabel(self)
        # self.label.resize(self.disply_width, self.display_height)
        # self.pushButton.clicked.connect()

    # def run(self):
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_pixmap_signal1.connect(self.update_image1)
        self.thread.change_pixmap_signal2.connect(self.update_image2)
        self.thread.change_pixmap_signal3.connect(self.update_image3)
        # l=QVBoxLayout(self)
        # self.labele=QLabel("0", self)
        # l.addWidget(self.labele)

        # self.thread=VideoThread()
        self.thread.update.connect(self.type_text.setText)
        self.thread.update1.connect(self.lp_text.setText)
        self.thread.update2.connect(self.character_text.setText)
        # self.thread=Thread(self)
        # self.thread.update.connect(self.text.setText("lab"))
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


        

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.original_img.setPixmap(qt_img)
        
    def update_image1(self, cv_img):
        """Updates the image_label with a new opencv image"""
        
        qt_img1 = self.convert_cv_qt(cv_img)
        self.type_img.setPixmap(qt_img1)
        
    def update_image2(self, cv_img):
        """Updates the image_label with a new opencv image"""
        
        qt_img2 = self.convert_cv_qt(cv_img)
        self.lp_img.setPixmap(qt_img2)
        
    def update_image3(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img3 = self.convert_cv_qt(cv_img)
        self.character_img.setPixmap(qt_img3)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
"""
End of: 
Main class to add functionality of designed GUI
"""


"""
Start of:
Main function
"""


# Defining main function to be run
def main():
    # Initializing instance of Qt Application
    # app = QtWidgets.QApplication(sys.argv)

    # # Initializing object of designed GUI
    # window = MainApp()

    # # Showing designed GUI
    # window.show()

    # # Running application
    # app.exec_()
    app = QApplication(sys.argv)
    a = MainApp()
    a.show()
    sys.exit(app.exec_())


"""
End of: 
Main function
"""


# Checking if current namespace is main, that is file is not imported
if __name__ == '__main__':
    # Implementing main() function
    main()

















# net = cv2.dnn.readNet(...)
# .
# .
# .
# del net