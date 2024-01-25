import sqlite3
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QStackedWidget
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QFormLayout


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


class FirstWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Первый экран (вход, регистрация, выход).ui", self)
        self.Exit.clicked.connect(sys.exit)


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
    registr.Back_Button.clicked.connect(lambda: w.setCurrentIndex(0))

    w.show()
    sys.exit(app.exec())