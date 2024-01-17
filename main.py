import csv
import xlsxwriter
import json

with open('headers.json', encoding='UTF-8') as file:
    headers = json.load(file)
    print(headers)


def from_dat(file_dir):
    content = pd.read_csv(file_dir, delim_whitespace=True, dtype=int, header=None)
    content.columns = headers
    print(content)


'''from_dat('forecasts/20046.dat')'''