import json
import sqlite3
from tqdm import tqdm
import pandas as pd
from process.process import *
import urllib.request

with open('process/headers.json', encoding='UTF-8') as file:
    headers = json.load(file)


def from_dat(file_dir, decode='process/Srok8c.ddl', is_url=False):
    all_, use_ = delim(decode)
    if is_url:
        f_read = urllib.request.urlopen(file_dir)
    else:
        f = open(file_dir)
        f_read = f.readlines()
    data = []
    data_type = [None] * len(all_)
    for i in tqdm(f_read):
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
    if not is_url:
        f.close()
    return pd.DataFrame(data, columns=headers.keys()), data_type


def from_csv(file_dir):
    df = pd.read_csv(file_dir)
    return df


def from_xlsx(file_dir):
    df = pd.read_excel(file_dir)
    return df


def column_sql(db_name, dir='database/temporary.db', target=None):
    conn = sqlite3.connect(dir)
    if target is None:
        result = pd.read_sql_query(f'''SELECT * FROM "{db_name}"
        ORDER BY ГОД, МЕСЯЦ, ДЕНЬ, ВРЕМЯМЕ''', conn)
    else:
        result = pd.read_sql_query(f'''SELECT {target} FROM "{db_name}"
        ORDER BY ГОД, МЕСЯЦ, ДЕНЬ, ВРЕМЯМЕ''', conn)

    # Закрываем соединение
    conn.close()

    return result


# Функция для записи данных в CSV файл
def to_csv(location_name, file_path):
    if file_path[-1] == '/':
        file_path += location_name + '.csv'
    elif '.' not in file_path:
        file_path += '.csv'
    df = column_sql(location_name)
    df.to_csv(file_path, index=False, encoding='UTF-8')
    print(f"Данные успешно записаны в {file_path}.")


# Функция для записи данных в Excel файл
def to_xlsx(location_name, file_path):
    if file_path[-1] == '/':
        file_path += location_name + '.xlsx'
    elif '.' not in file_path:
        file_path += '.xlsx'
    df = column_sql(location_name)
    df.to_excel(file_path, index=False)
    print(f"Данные успешно записаны в {file_path}.")


def to_sql(db, name='meteorological_data', header=None, types=None):
    if header is None:
        header = list(headers.keys())
    le = len(db.columns)
    if types is None:
        types = ['REAL'] * (len(headers) if headers else le)
    head_line = ', '.join(header[i] + ' ' + types[i] for i in range(le))

    conn = sqlite3.connect('database/temporary.db')
    c = conn.cursor()
    c.execute(f'CREATE TABLE IF NOT EXISTS "{name}"({head_line})')
    for index, value in tqdm(db.iterrows()):
        c.execute(f'INSERT INTO "{name}" VALUES({",".join(["?"] * le)})', list(value))

    # Сохраняем изменения
    conn.commit()

    # Закрываем соединение
    conn.close()


if __name__ == '__main__':
    print(column_sql('20087'))