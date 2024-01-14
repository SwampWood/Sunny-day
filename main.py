import pandas as pd
import csv
import xlsxwriter


headers = ['Station', 'YearGr', 'MonthGr', 'DayGr', 'PeriodGr', 'Year', 'Month', 'Day', 'Period', 'PeriodPDZV',
           'Time', 'GMT', 'IndArch', 'ViewHor', 'Ind(ViewHor)', 'Sign(ViewHor)', 'CloudCount', 'Ind(CloudCount)',
           'BotCount', 'Ind(BotCount)', 'TopForm', 'Ind(TopForm)', 'MidForm', 'Ind(MidForm)', 'TopPrForm',
           'Ind(TopPrForm)'] + list(map(str, range(49)))
print(len(headers))


def from_dat(file_dir):
    content = pd.read_csv(file_dir, delim_whitespace=True, dtype=int, header=None)
    content.columns = headers
    print(content)


from_dat('forecasts/20046.dat')