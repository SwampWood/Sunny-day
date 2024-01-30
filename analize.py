import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
from format import column_sql, headers
from prediction import LargeDataset, sort_


def check_stat(data):
    result = adfuller(data)
    return result[1] < 0.05


def characteristics(df):
    mean = df.mean(axis=0, skipna=True)
    mode = df.mode().iloc[0]
    median = df.median(axis=0, skipna=True)
    max_ = df.max(axis=0, skipna=True)
    min_ = df.min(axis=0, skipna=True)
    df_dup = pd.DataFrame([mean, mode, median, max_, min_],
                          index=('Среднее', 'Мода', 'Медиана', 'Максимум', 'Минимум'))
    return df_dup.drop(columns=['ДАТАВРЕМЯ'])


def combine_seas_cols(df):
    sd = seasonal_decompose(df, freq=2920)
    df['observed'] = sd.observed
    df['residual'] = sd.resid
    df['seasonal'] = sd.seasonal
    df['trend'] = sd.trend


def mround(x, m=5):
    return int(m * round(float(x) / m))


def plot_components(df):
    df_axis = df.fillna(0)
    ymin = mround(np.min([df_axis.observed, df_axis.trend, df_axis.seasonal]), 5)
    ymax = mround(np.max([df_axis.observed, df_axis.trend, df_axis.seasonal]), 5)
    ymin -= 5
    ymax += 5
    objects = np.array([df_axis.observed, df_axis.trend, df_axis.seasonal]).T

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 1, 1)
    plt.ylim(ymin, ymax)
    plt.legend(plt.plot(df.index, objects), ('Наблюдения', 'Тренд', 'Сезонность'))

    plt.show()


if __name__ == '__main__':
    whole_data = LargeDataset(column_sql('database/temporary.db'), params=tuple(headers.keys()))
    df = whole_data.to_pd()
    df_temp = df[['ДАТАВРЕМЯ', 'ДАВЛАУММ']]
    # df['ТЕМВОЗДМ'] = df['ТЕМВОЗДМ'] / df['ДАТАВРЕМЯ'].dt.day
    df_cop = sort_(df_temp)
    combine_seas_cols(df_cop)
    print(characteristics(df))
    plot_components(df_cop)