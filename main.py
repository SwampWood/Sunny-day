import sqlite3
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QDateEdit
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QFormLayout


class FirstWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("first_screen.ui", self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    exe = FirstWindow()
    exe.show()
    sys.exit(app.exec())