import sqlite3
import sys
import ctypes
import analize

import pandas as pd
import datetime as dt

from PyQt5 import uic, QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QStackedWidget, QMainWindow, QFileDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from time import sleep
from format import *
from current import get_current
from prediction import LargeDataset, sort_, format_predictions


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.index)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row()][index.column()])
        return None

    def headerData(self, rowcol, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[rowcol]
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[rowcol]
        return None


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


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)

    def check_time(self):
        while True:
            if dt.datetime.now().minute % 18 == 0:
                title = 'Уведомление'
                message = 'Внимание! В течении часа ожидается ураган со скоростью ветра до 10 м/с. '\
                          'Будьте осторожны и не покидайте помещения'
                ctypes.windll.user32.MessageBoxW(0, message, title)
                sleep(600)
            else:
                sleep(60)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/Главное меню.ui", self)
        self.con = sqlite3.connect('database/temporary.db')
        self.cursor = self.con.cursor()
        self.df = None
        self.df2 = None

        self.Exit.clicked.connect(lambda: w.setCurrentIndex(0))
        self.Exit_2.clicked.connect(lambda: w.setCurrentIndex(0))
        self.Exit_3.clicked.connect(lambda: w.setCurrentIndex(0))
        self.Exit_4.clicked.connect(lambda: w.setCurrentIndex(0))
        self.Load.clicked.connect(self.open_file)
        self.Export.clicked.connect(self.new_file)
        self.checkBox.stateChanged.connect(self.show_url)
        self.checkBox_2.stateChanged.connect(self.show_loc)
        self.lineEdit_2.setVisible(False)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.verticalLayout_2.addWidget(self.canvas)

        self.Error2.setVisible(False)
        self.Error2.setStyleSheet("color: red")

        self.comboBox_2.currentTextChanged.connect(self.change_options)
        self.comboBox_4.currentTextChanged.connect(self.change_options2)
        self.comboBox_3.currentTextChanged.connect(self.plot)
        self.comboBox_8.currentTextChanged.connect(self.predict)
        self.pushButton_2.clicked.connect(self.set_current)
        self.dateEdit.dateChanged.connect(self.plot)
        self.dateEdit_2.dateChanged.connect(self.plot)
        self.set_options()
        self.ShowGraph.clicked.connect(self.show_graph)

    def show_url(self):
        self.lineEdit_2.setVisible(self.checkBox.isChecked())

    def open_file(self):
        self.Error2.setVisible(False)
        self.Error2.setStyleSheet("color: red")
        if self.checkBox.isChecked():
            url = self.lineEdit_2.text()
            type_ = url.split('.')[-1].upper()
            if url:
                data = None
                types = None
                if type_ == 'DAT':
                    data, types = from_dat(url, is_url=True)
                elif type_ == 'CSV':
                    data = from_csv(url)
                elif type_ == 'XLSX':
                    data = from_xlsx(url)
                else:
                    self.Error2.setVisible(True)
                    self.Error2.setText("Неверный формат файла")
                    pass
                name = url[url.rfind('/') + 1:url.rfind('.')]
                to_sql(data, name=name, header=data.columns.values.tolist(), types=types)
                self.comboBox_2.addItem(name)
                self.comboBox_4.addItem(name)
                self.comboBox_6.addItem(name)
                self.Error2.setVisible(True)
                self.Error2.setStyleSheet("color: green")
                self.Error2.setText("Файл успешно загружен")
        else:
            filename, type_ = QFileDialog.getOpenFileName(self, "Open File", ".",
                                                          "Text Files (*.dat);;Tables(*.csv);;Tables(*.xlsx);;All Files (*)")
            print(filename, type_)
            if filename:
                data = None
                types = None
                if type_ == '*':
                    if filename[filename.rfind('.') + 1:] not in ['dat', 'csv', 'xslx']:
                        self.Error2.setVisible(True)
                        self.Error2.setText("Недопустимый формат файла")
                        pass
                    elif filename[filename.rfind('.') + 1:] == 'dat':
                        data, types = from_dat(filename)
                    elif filename[filename.rfind('.') + 1:] == 'csv':
                        data = from_csv(filename)
                    elif filename[filename.rfind('.') + 1:] == 'xlsx':
                        data = from_xlsx(filename)
                else:
                    try:
                        if type_ == 'Text Files (*.dat)':
                            data, types = from_dat(filename)
                        elif type_ == 'Tables(*.csv)':
                            data = from_csv(filename)
                        elif type_ == 'Tables(*.xlsx)':
                            data = from_xlsx(filename)
                    except Exception:
                        self.Error2.setVisible(True)
                        self.Error2.setText("Непредвиденная ошибка загрузки")
                        pass
                name = filename[filename.rfind('/') + 1:filename.rfind('.')]
                to_sql(data, name=name, header=data.columns.values.tolist(), types=types)
                self.comboBox_0.addItem(name)
                self.comboBox_2.addItem(name)
                self.comboBox_4.addItem(name)
                self.comboBox_6.addItem(name)
                self.Error2.setVisible(True)
                self.Error2.setStyleSheet("color: green")
                self.Error2.setText("Файл успешно загружен")

            else:
                self.Error2.setVisible(True)
                self.Error2.setText("Ошибка выполнения")

    def new_file(self):
        name = self.comboBox_0.currentText().lower() + '.' + self.comboBox.currentText().lower()
        filename, type_ = QFileDialog.getSaveFileName(self, "Open File", "./" + name, "All Files (*)")
        if filename:
            if self.comboBox.currentText() == 'CSV':
                to_csv(self.comboBox_0.currentText(), filename)
            elif self.comboBox.currentText() == 'XLSX':
                to_xlsx(self.comboBox_0.currentText(), filename)
            self.Error2.setStyleSheet("color: green")
            self.Error2.setText("Экспорт успешно завершен")
        else:
            self.Error2.setVisible(True)
            self.Error2.setText("Ошибка экспорта")

    def set_options(self):
        res = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        for (i,) in res:
            self.comboBox_0.addItem(i)
            self.comboBox_2.addItem(i)
            self.comboBox_4.addItem(i)
            self.comboBox_6.addItem(i)

    # 1 экран
    def change_options(self):
        self.comboBox_3.clear()
        self.df = LargeDataset(column_sql(self.comboBox_2.currentText()), params=tuple(headers.keys())).to_pd()
        if self.comboBox_2.currentText().isdigit():
            des = LargeDataset.required
            for j in des[4:]:
                self.comboBox_3.addItem(j)
        else:
            des = self.cursor.execute(f'SELECT * FROM "{self.comboBox_2.currentText()}"').description
            for j in des:
                self.comboBox_3.addItem(j[0])

    def plot(self):
        self.figure.clear()

        ax = self.figure.add_subplot(111)

        ax.clear()

        date1 = self.dateEdit.date().toString('yyyy-MM-dd')
        date2 = self.dateEdit_2.date().toString('yyyy-MM-dd')
        ax.plot(self.df.loc[date1:date2].index.values,
                self.df.loc[date1:date2][self.comboBox_3.currentText()].values)

        self.canvas.draw()

    # 2 экран
    def change_options2(self):
        self.comboBox_5.clear()
        self.df2 = LargeDataset(column_sql(self.comboBox_4.currentText()),
                                params=tuple(headers.keys())).to_pd().reset_index()
        if self.comboBox_4.currentText().isdigit():
            des = LargeDataset.required
            for j in des[4:]:
                self.comboBox_5.addItem(j)
        else:
            des = self.cursor.execute(f'SELECT * FROM "{self.comboBox_4.currentText()}"').description
            for j in des:
                self.comboBox_5.addItem(j[0])
        model = PandasModel(analize.characteristics(self.df2))
        self.tableView.setModel(model)

    def show_graph(self):
        df3 = sort_(self.df2[['ДАТАВРЕМЯ', self.comboBox_5.currentText()]])
        analize.combine_seas_cols(df3)
        analize.plot_components(df3)

    # 3 экран
    def predict(self):
        df = LargeDataset(column_sql(self.comboBox_6.currentText()), params=tuple(headers.keys())).to_pd().reset_index()
        df4 = format_predictions(df, self.comboBox_8.currentText())
        model = PandasModel(df4)
        self.tableView_2.setModel(model)

    # 4 экран
    def set_current(self):
        df = pd.DataFrame(get_current(self.lineEdit.text(), my=not self.checkBox_2.isChecked()))
        model = PandasModel(df)
        self.tableView_3.setModel(model)

    def show_loc(self):
        self.lineEdit.setVisible(self.checkBox.isChecked())
        self.lineEdit.clear()


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

    worker = Worker()
    thread = QtCore.QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.check_time)
    thread.start()

    w.show()
    sys.exit(app.exec())