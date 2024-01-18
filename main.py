import csv
import xlsxwriter
import json
import sqlite3
from process.process import *
import time

with open('process/headers.json', encoding='UTF-8') as file:
    headers = json.load(file)


def from_dat(file_dir, decode='Srok8c.ddl'):
    all_, use_ = delim(decode)
    with open(file_dir) as f:
        data = []
        data_type = [None] * len(all_)
        for i in f.readlines():
            i = i.strip()
            row = []
            cur = 0
            for j in range(len(all_)):
                cur += all_[j]
                idx = i[cur - all_[j]:cur]
                idx = idx.replace(' ', '')
                if data_type[j] is None and idx and '.' in idx:
                    data_type[j] = 'REAL'
                    row.append(float(idx))
                elif data_type[j] is None and idx:
                    data_type[j] = 'INT'
                    row.append(int(idx))
                else:
                    row.append('NaN')
            data.append(row)
    return data


def to_sql(db):
    conn = sqlite3.connect('temporary.db')
    c = conn.cursor()

    # Замените 'table_name' именем вашей таблицы
    c.execute("CREATE TABLE IF NOT EXISTS table_name (column1 TEXT, column2 REAL)")

    # Вставляем значения в таблицу
    values = [(u'John', 21.0), (u'Jane', 19.0)]

    for value in values:
        c.execute('INSERT INTO table_name VALUES(?,?)', value)

    # Сохраняем изменения
    conn.commit()

    # Закрываем соединение
    conn.close()

print(from_dat('forecasts/20046.dat'))