import pyodbc, urllib, sqlalchemy
import pandas as pd
import numpy as np

from time import time
from config import *
from db_config import password
from datetime import datetime
from windside_extract_data import interpolate_negative_voltage

if __name__ == '__main__':

    # input fpath

    input_fpath = osp.join(WINDSIDE_DATA_DIR, 'features.csv')

    # table and columns

    id_col = 'ID'
    timestamp_col = 'Date'
    header = 'Wind speed, # samples, Avg current, Min current, Max current, Avg voltage, Min voltage, Max voltage, Avg power, Min power, Max power'

    # select data

    chunk_size = None
    ts_step = 3600
    windspeed_step = 0.1

    if chunk_size is not None:
        df = pd.read_csv(input_fpath, chunksize=chunk_size)
        df = next(df)
    else:
        df = pd.read_csv(input_fpath)

    vals = df.values
    print(vals.shape, vals[0, 0], vals[-1, 0])

    voltage = vals[:, 3]

    print(len(np.where(voltage >= 0)[0]), len(np.where(voltage < 0)[0]))

    idx = np.where(voltage >= 0)[0]
    ids = vals[idx, 0]
    dates = vals[idx, 1]
    currents = vals[idx, 2]
    voltages = vals[idx, 3]
    windspeeds = vals[idx, 4]

    del vals

    str_last_date = pd.to_datetime(str(dates[-1])).strftime('%d/%m/%Y %H:%M:%S')
    print('Last date:', str_last_date)

    dt_dates = [pd.to_datetime(d) for d in dates]
    tss = np.array([d.timestamp() for d in dt_dates])

    ts_min = np.min(tss)
    ts_max = np.max(tss)

    n_ts = int((ts_max - ts_min) / ts_step) + 1

    print('Number of time intervals:', n_ts)

    lines = []

    offset = 0
    last_possible = windspeed_step

    for ti in np.arange(n_ts):

        t = ts_min // ts_step * ts_step + ti * ts_step

        idx_t = np.where((tss[int(offset) : int(offset + 2 * ts_step)] >= t) & (tss[int(offset) : int(offset + 2 * ts_step)] < t + ts_step))[0]

        t_start_str = datetime.strftime(datetime.utcfromtimestamp(t), '%Y/%m/%d %H:%M:%S')
        t_stop_str = datetime.strftime(datetime.utcfromtimestamp(t + ts_step - 1), '%Y/%m/%d %H:%M:%S')

        print(t_start_str, '-', t_stop_str, len(idx_t))

        lines.append(f'Interval: {t_start_str} - {t_stop_str}\n')
        lines.append(header)

        if len(idx_t) > 0:

            idx_t = idx_t + offset

            voltage_t = voltages[idx_t]
            current_t = currents[idx_t]
            power_t = current_t * np.abs(voltage_t)

            windspeed_t = windspeeds[idx_t]
            windspeed_t_min = np.min(windspeed_t)
            windspeed_t_max = np.max(windspeed_t)

            n_ws = int((windspeed_t_max - windspeed_t_min) / windspeed_step)

            for wi in np.arange(n_ws):

                w = windspeed_t_min // windspeed_step * windspeed_step + wi * windspeed_step
                idx_tw = np.where((windspeed_t >= w) & (windspeed_t <= w + windspeed_step))[0]

                n_power = len(idx_tw)
                if n_power > 0:
                    current_mean = np.mean(current_t[idx_tw])
                    current_min = np.min(current_t[idx_tw])
                    current_max = np.max(current_t[idx_tw])
                    voltage_mean = np.mean(voltage_t[idx_tw])
                    voltage_min = np.min(voltage_t[idx_tw])
                    voltage_max = np.max(voltage_t[idx_tw])
                    power_mean = np.mean(power_t[idx_tw])
                    power_min = np.min(power_t[idx_tw])
                    power_max = np.max(power_t[idx_tw])
                else:
                    current_mean = None
                    current_min = None
                    current_max = None
                    voltage_mean = None
                    voltage_min = None
                    voltage_max = None
                    power_mean = None
                    power_min = None
                    power_max = None

                lines.append(f'{w:.1f} - {w + windspeed_step:.1f}, {n_power}, {current_mean}, {current_min}, {current_max}, {voltage_mean}, {voltage_min}, {voltage_max}, {power_mean}, {power_min}, {power_max}')

            offset = idx_t[-1]

        lines.append('')

    with open('windside_summary.txt', 'w') as f:
        f.writelines('\n'.join(lines))