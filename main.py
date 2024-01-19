import sqlite3
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QStackedWidget
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QFormLayout


class Registration(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Регистрация.ui", self)


class Authorization(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Вход.ui", self)


class FirstWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Первый экран (вход, регистрация, выход).ui", self)
        self.Exit_Button.clicked.connect(sys.exit)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    exe = FirstWindow()
    registr = Registration()
    auth = Authorization()
    w = QStackedWidget()
    w.addWidget(exe)
    w.addWidget(auth)
    w.addWidget(registr)

    exe.Entrance_Button.clicked.connect(lambda: w.setCurrentIndex(1))
    exe.Registration_Button.clicked.connect(lambda: w.setCurrentIndex(2))
    auth.Back_Button.clicked.connect(lambda: w.setCurrentIndex(0))
    # registr.Back_Button.clicked.connect(lambda: w.setCurrentIndex(0))

    w.show()
    sys.exit(app.exec())