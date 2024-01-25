import sqlite3

def registration(email, login, password):
    with sqlite3.connect('k.db') as k:
        cur = k.cursor()
        cur.execute(f"SELECT login FROM entrance")
        a = cur.fetchall()
        li = [str(i).split("'")[1] for i in a] # Это список всех логинов
        if login in li:  # Проверка на повтор логинов
            print('Такой логин уже есть')
        else:
            params = (email, login, password)
            cur.execute("INSERT INTO entrance VALUES(?, ?, ?)", params)

def entrance(email, login, password):
    with sqlite3.connect('k.db') as k:
        cur = k.cursor()
        cur.execute(f"SELECT login FROM entrance")
        a = cur.fetchall()
        li = [str(i).split("'")[1] for i in a]  # Это список всех логинов
        print(li)
        if login not in li: #Чтобы код не выпадал с ошибкой, проверяем, существует ли такой логин
            print('Такого пользователя не существует')
        else:
            cur.execute(f"SELECT * FROM entrance WHERE login == '{login}'")
            result = list(cur.fetchone())
            print(result)
            if result[2] == password: #Тут сравнение с паролем при вводе и том, что в базе данных
                print('Пароль верный')
            else:
                print('Пароль неверный')
