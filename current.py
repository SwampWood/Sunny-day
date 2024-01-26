import requests


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


def get_current(pos):
    lon, lat = map(float, get_toponym(pos).split())
    params = {
        "lat": lat,
        "lon": lon,
        "appid": "2bfc39f739b1ca67626578f67317cf41",
        "exclude": "minutely,hourly,daily",
        "units": "metric",
        "lang": "ru"
    }
    resp = requests.get('http://api.openweathermap.org/data/2.5/weather?', params=params)
    return resp.json()


if __name__ == '__main__':
    print(get_current('Москва'))