import pyodbc, urllib, sqlalchemy
import pandas as pd
import numpy as np

from time import time
from config import *
from db_config import password
from datetime import datetime


if __name__ == '__main__':

    # input fpath

    input_fpath = osp.join(WINDSIDE_DATA_DIR, 'features.csv')

    # db access

    server = 'tcp:jyusqlserver.database.windows.net'
    database = 'IoTSQL'
    driver = '{SQL Server}'
    username = 'jyusqlserver'
    db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

    # table and columns

    #table = 'WindsideHourData'
    table = 'Windside10minData'

    id_col = 'ID'
    start_date_col = 'StartDate'
    start_timestamp_col = 'StartTimestamp'
    end_date_col = 'EndDate'
    end_timestamp_col = 'EndTimestamp'
    last_id_col = 'LastId'
    n_col = 'NumberOfSamples'

    # select data

    chunk_size = None
    n_samples = chunk_size

    #ts_step = 60 * 60
    ts_step = 60 * 10

    if chunk_size is not None:
        df = pd.read_csv(input_fpath, chunksize=chunk_size)
        df = next(df)
    else:
        df = pd.read_csv(input_fpath)

    vals = df.values
    print(vals.shape, vals[0, 0], vals[-1, 0])

    voltages = vals[:, 3]

    print(len(np.where(voltages > 0)[0]), len(np.where(voltages < 0)[0]))

    idx = np.where(voltages >= 0)[0]

    ids = vals[idx, 0]
    dates = vals[idx, 1]
    currents = vals[idx, 2]
    voltages = voltages[idx]
    power = currents * np.abs(voltages)
    windspeeds = vals[idx, 4]
    unixtimes = vals[idx, 5]

    t_start = time()

    last_id = ids[-1]
    #str_last_date = pd.to_datetime(str(dates[-1])).strftime('%d/%m/%Y %H:%M:%S')
    str_last_date = datetime.strptime(str(dates[-1]), '%d/%m/%Y %H:%M:%S').strftime('%d/%m/%Y %H:%M:%S')

    timestamp_min = unixtimes[0]
    timestamp_max = unixtimes[-1]
    n_timestamps = int((timestamp_max - timestamp_min) / ts_step) + 1

    print(dates[-1], str_last_date, n_timestamps)

    values = []

    offset, last_offset = 0, 0

    for i in range(n_timestamps):

        s_ts = timestamp_min // ts_step * ts_step + i * ts_step
        s_ts_dt = datetime.fromtimestamp(s_ts).strftime('%d/%m/%Y %H:%M:%S')
        e_ts = s_ts + ts_step
        e_ts_dt = datetime.fromtimestamp(e_ts).strftime('%d/%m/%Y %H:%M:%S')

        idx = offset + np.where((unixtimes[offset:offset+ts_step * 2] >= s_ts) & (unixtimes[offset:offset+ts_step * 2] < e_ts))[0]
        n_power = len(idx)

        print(s_ts_dt, e_ts_dt, n_power)

        if n_power > 0:

            current_avg = np.mean(currents[idx])
            current_min = np.min(currents[idx])
            current_max = np.max(currents[idx])
            voltage_avg = np.mean(voltages[idx])
            voltage_min = np.min(voltages[idx])
            voltage_max = np.max(voltages[idx])
            power_avg = np.mean(power[idx])
            power_min = np.min(power[idx])
            power_max = np.max(power[idx])
            windspeed_avg = np.mean(windspeeds[idx])
            windspeed_min = np.min(windspeeds[idx])
            windspeed_max = np.max(windspeeds[idx])

            values.append([
                s_ts_dt,        # first date in string format
                s_ts,           # first date in unix format
                e_ts_dt,        # first date in string format
                e_ts,           # first date in unix format
                ids[idx[-1]],   # last id
                n_power,        # number of samples
                current_avg,
                current_min,
                current_max,
                voltage_avg,
                voltage_min,
                voltage_max,
                power_avg,
                power_min,
                power_max,
                windspeed_avg,
                windspeed_min,
                windspeed_max
            ])

            last_offset = offset
            offset = idx[-1]

        else:
            #offset = np.where(unixtimes <= e_ts)[0][-1]
            #offset = last_offset
            offset = last_offset + np.where(unixtimes[last_offset:] <= e_ts)[0][-1]
            print(offset)

    cols = [
        'AverageCurrent',
        'MinimalCurrent',
        'MaximalCurrent',

        'AverageVoltage',
        'MinimalVoltage',
        'MaximalVoltage',

        'AveragePower',
        'MinimalPower',
        'MaximalPower',

        'AverageWindSpeed',
        'MinimalWindSpeed',
        'MaximalWindSpeed'
    ]

    sqlalchemy_id_col = sqlalchemy.Column(id_col, sqlalchemy.INT, primary_key=True)
    sqlalchemy_start_date_col = sqlalchemy.Column(start_date_col, sqlalchemy.String)
    sqlalchemy_start_timestamp_col = sqlalchemy.Column(start_timestamp_col, sqlalchemy.Float)
    sqlalchemy_end_date_col = sqlalchemy.Column(end_date_col, sqlalchemy.String)
    sqlalchemy_end_timestamp_col = sqlalchemy.Column(end_timestamp_col, sqlalchemy.Float)
    sqlalchemy_last_id_col = sqlalchemy.Column(last_id_col, sqlalchemy.INT)
    sqlalchemy_n_col = sqlalchemy.Column(n_col, sqlalchemy.INT)
    sqlalchemy_cols = [sqlalchemy.Column(col, sqlalchemy.Float) for col in cols]

    params = urllib.parse.quote_plus(db_connection_str)
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    meta = sqlalchemy.MetaData(engine)
    table_pointer = sqlalchemy.Table(
        table,
        meta,
        sqlalchemy_id_col,
        sqlalchemy_start_date_col,
        sqlalchemy_start_timestamp_col,
        sqlalchemy_end_date_col,
        sqlalchemy_end_timestamp_col,
        sqlalchemy_last_id_col,
        sqlalchemy_n_col,
        *sqlalchemy_cols
    )
    meta.create_all(checkfirst=True)
    conn = engine.connect()

    query = f"insert into {table} ({','.join([start_date_col, start_timestamp_col, end_date_col, end_timestamp_col, last_id_col, n_col] + cols)}) values ({','.join(['?' for _ in [start_date_col, start_timestamp_col, end_date_col, end_timestamp_col, last_id_col, n_col] + cols])})"

    print(query, len(values), values[-1])

    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, values)

    # Uncomment for the update test:

    #query = "update WindsidePowerSummaryAllTime set NumberOfSamples = 312928,TotalPower = 5417155.441204475,MinimalPower = 0.0,MaximalPower = 153.87377275699998 where WindSpeedLeft = 2.6;"
    #query = f"update {table} set WindSpeedLeft = ?,WindSpeedRight = ? where WindSpeedLeft = ?;"
    #values = [0.0, 0.123, 0.0]
    #print(query)
    #with pyodbc.connect(db_connection_str) as conn:
    #   with conn.cursor() as cursor:
    #       cursor.execute(query, values)
    #       #cursor.execute(query)