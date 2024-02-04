import requests
import geocoder
import numpy as np
import datetime


needed_params = ['Дата и время', 'Локация', 'Температура', 'Давление', 'Влажность',
                 'Скорость ветра', 'Направление ветра', 'Осадки (мм)', 'Облачность']


def get_toponym(pos):
    geocoder_api_server = "http://geocode-maps.yandex.ru/1.x/"
    geocoder_params = {
        "apikey": "40d1649f-0493-4b70-98ba-98533de7710b",
        "geocode": pos,
        "format": "json"}
    response = requests.get(geocoder_api_server, params=geocoder_params)
    if not response:
        print(response, response.url)
        return
    json_object = response.json()
    amount_results = json_object["response"]["GeoObjectCollection"]["metaDataProperty"]["GeocoderResponseMetaData"]
    if not int(amount_results["found"]):
        print(f"request: {amount_results['request']}; found; {amount_results['found']}")
        return
    toponym = json_object["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]
    address_point = toponym["Point"]["pos"]
    return address_point


def get_reverse_toponym(pos):
    geocoder_api_server = "http://geocode-maps.yandex.ru/1.x/"
    geocoder_params = {
        "apikey": "40d1649f-0493-4b70-98ba-98533de7710b",
        "geocode": ','.join(pos),
        "format": "json"}
    response = requests.get(geocoder_api_server, params=geocoder_params)
    if not response:
        print(response, response.url)
        return
    json_object = response.json()
    amount_results = json_object["response"]["GeoObjectCollection"]["metaDataProperty"]["GeocoderResponseMetaData"]
    if not int(amount_results["found"]):
        print(f"request: {amount_results['request']}; found; {amount_results['found']}")
        return
    toponym = json_object["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]
    address_name = toponym["metaDataProperty"]["GeocoderMetaData"]["text"]
    return address_name


def get_user():
    g = geocoder.ip('me')
    return g.latlng


def get_current(pos, my=False):
    if my:
        lat, lon = get_user()
    else:
        lon, lat = map(float, get_toponym(pos).split())
    params = {
        "lat": lat,
        "lon": lon,
        "appid": "2b" + "fc39f" + "739b1ca676" + "7562"[::-1] + "8f673" + "17cf41",
        "exclude": "minutely,hourly,daily",
        "units": "metric",
        "lang": "ru"
    }
    resp = requests.get('http://api.openweathermap.org/data/2.5/weather?', params=params).json()
    date = datetime.datetime.utcfromtimestamp(resp['dt'] + resp['timezone']).strftime('%Y-%m-%d %H:%M:%S')

    print(resp)
    res = {
        needed_params[0]: np.array([date]),
        needed_params[1]: np.array([resp['name']]),
        needed_params[2]: np.array([resp['main']['temp']]),
        needed_params[3]: np.array([resp['main']['pressure']]),
        needed_params[4]: np.array([resp['main']['humidity']]),
        needed_params[5]: np.array([resp['wind']['speed']]),
        needed_params[6]: np.array([resp['wind']['deg']]),
        needed_params[7]: np.array([resp.get('snow', {'1h': 0})['1h'] + resp.get('rain', {'1h': 0})['1h']]),
        needed_params[8]: np.array([resp['clouds']['all']]),
    }
    return res


if __name__ == '__main__':
    print(get_user())
    print(get_current('Москва'))
