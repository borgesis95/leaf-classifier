
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from Main import training
class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(512, 296)
        
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(50, 220, 421, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.run_train)

        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(40, 50, 101, 21))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(["Squeezenet","ResNet","vgg16"])

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 30, 91, 16))
        self.label.setObjectName("label")

        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(170, 50, 121, 21))
        self.textEdit.setObjectName("textEdit")

        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(170, 30, 47, 13))
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Trainer APP"))
        self.pushButton.setText(_translate("Dialog", "Start training"))
        self.label.setText(_translate("Dialog", "Choose Model"))
        self.label_2.setText(_translate("Dialog", "Epochs:"))

    def run_train(self):
        modelName = self.comboBox.currentText()
        epochs = int(self.textEdit.toPlainText())
        
        training(modelName,False,True,epochs)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
