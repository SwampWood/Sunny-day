import sqlite3
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QStackedWidget, QMainWindow
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Registration(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Регистрация.ui", self)
        self.inputs = {'Почта': self.Login, 'Логин': self.lineEdit,
                       'Пароль': self.Password, 'Повтор пароля': self.lineEdit_2}
        self.Registration_Button.clicked.connect(self.registration)
        self.label_Errors.setStyleSheet("color: red")
        self.label_Errors.setVisible(False)

    def registration(self):
        email = self.inputs['Почта'].text()
        login = self.inputs['Логин'].text()
        password = self.inputs['Пароль'].text()
        password_dup = self.inputs['Повтор пароля'].text()
        if not (email and login and password and password_dup):
            self.label_Errors.setVisible(True)
            self.label_Errors.setText('Не все поля заполнены')
        elif password != password_dup:
            self.label_Errors.setVisible(True)
            self.label_Errors.setText('Пароли не совадают')
        elif '@' not in email:
            self.label_Errors.setVisible(True)
            self.label_Errors.setText('Неправильный формат почты')
        else:
            self.label_Errors.setVisible(False)
            with sqlite3.connect('database/users.db') as k:
                cur = k.cursor()
                cur.execute(f"SELECT login FROM entrance")
                a = cur.fetchall()
                li = [str(i).split("'")[1] for i in a]  # Это список всех логинов
                if login in li:  # Проверка на повтор логинов
                    self.label_Errors.setVisible(True)
                    self.label_Errors.setText('Такой логин уже есть')
                else:
                    params = (email, login, password)
                    cur.execute("INSERT INTO entrance VALUES(?, ?, ?)", params)
                    w.setCurrentIndex(3)


class Authorization(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Вход.ui", self)
        self.inputs = {'Логин': self.Login, 'Пароль': self.Password}
        self.Entrance_Button.clicked.connect(self.entrance)
        self.label_Errors.setStyleSheet("color: red")
        self.label_Errors.setVisible(False)

    def entrance(self):
        login = self.inputs['Логин'].text()
        password = self.inputs['Пароль'].text()
        if not (login and password):
            self.label_Errors.setVisible(True)
            self.label_Errors.setText('Не все поля заполнены')
        else:
            self.label_Errors.setVisible(False)
            with sqlite3.connect('database/users.db') as k:
                cur = k.cursor()
                cur.execute(f"SELECT login FROM entrance")
                a = cur.fetchall()
                li = [str(i).split("'")[1] for i in a]  # Это список всех логинов
                print(li)
                if login not in li:  # Чтобы код не выпадал с ошибкой, проверяем, существует ли такой логин
                    self.label_Errors.setVisible(True)
                    self.label_Errors.setText('Такого пользователя не существует')
                else:
                    cur.execute(f"SELECT * FROM entrance WHERE login == '{login}'")
                    result = list(cur.fetchone())
                    print(result)
                if result[2] != password:  # Тут сравнение с паролем при вводе и том, что в базе данных
                    self.label_Errors.setVisible(True)
                    self.label_Errors.setText('Пароль неверный')
                else:
                    self.label_Errors.setVisible(True)
                    self.label_Errors.setText('Пароль верный')
                    w.setCurrentIndex(3)


class FirstWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Первый экран (вход, регистрация, выход).ui", self)
        self.Exit.clicked.connect(sys.exit)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Главное меню.ui", self)
        self.Exit.clicked.connect(lambda: w.setCurrentIndex(0))
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.verticalLayout_2.addWidget(self.canvas)
        self.Error1.setVisible(False)
        self.Error2.setVisible(False)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", ".", "Text Files (*.txt);;All Files (*)")
        if filename:
            with open(filename, 'r') as file:
                contents = file.read()
            print(contents)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    exe = FirstWindow()
    registr = Registration()
    auth = Authorization()
    main = MainWindow()
    w = QStackedWidget()
    w.addWidget(exe)
    w.addWidget(auth)
    w.addWidget(registr)
    w.addWidget(main)

    exe.Entrance_Button.clicked.connect(lambda: w.setCurrentIndex(1))
    exe.Registration_Button.clicked.connect(lambda: w.setCurrentIndex(2))
    auth.Back_Button.clicked.connect(lambda: w.setCurrentIndex(0))
    registr.Back_Button.clicked.connect(lambda: w.setCurrentIndex(0))

    w.show()
    sys.exit(app.exec())