import csv
import xlsxwriter
import json
import sqlite3
from tqdm import tqdm
import os
from process.process import *

with open('process/headers.json', encoding='UTF-8') as file:
    headers = json.load(file)


def from_dat(file_dir, decode='process/Srok8c.ddl'):
    all_, use_ = delim(decode)
    with open(file_dir) as f:
        data = []
        data_type = [None] * len(all_)
        for i in tqdm(f.readlines()):
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
                elif not idx:
                    row.append('NULL')
                elif data_type[j] == 'REAL':
                    row.append(float(idx))
                elif data_type[j] == 'INT':
                    row.append(int(idx))
            data.append(row)
    for i in range(len(data_type)):
        if data_type[i] is None:
            data_type[i] = 'STRING'
    return data, data_type


def to_sql(db, header=None, types=None):
    if header is None:
        header = list(headers.keys())
    le = len(db[0])
    if types is None:
        types = ['REAL'] * (len(headers) if headers else le)
    head_line = ', '.join(header[i] + ' ' + types[i] for i in range(le))

    conn = sqlite3.connect('database/temporary.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS meteorological_data (' + head_line + ')')
    for value in db:
        c.execute(f'INSERT INTO meteorological_data VALUES({",".join(["?"] * le)})', value)

    # Сохраняем изменения
    conn.commit()

    # Закрываем соединение
    conn.close()


if __name__ == '__main__':
    try:
        os.remove('database/temporary.db')
    except Exception:
        os.remove('database/temporary.db')
    db, data_types = from_dat('forecasts/20046.dat')
    to_sql(db, types=data_types)