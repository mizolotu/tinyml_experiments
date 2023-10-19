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

    # db access

    server = 'tcp:jyusqlserver.database.windows.net'
    database = 'IoTSQL'
    driver = '{SQL Server}'
    username = 'jyusqlserver'
    db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

    # table and columns

    table = 'WindsidePowerSummaryAllTime'
    id_col = 'ID'
    timestamp_col = 'Date'

    # select data

    chunk_size = None
    n_samples = chunk_size
    windspeed_min = 0
    windspeed_max = 40
    windspeed_step = 0.1

    if chunk_size is not None:
        df = pd.read_csv(input_fpath, chunksize=chunk_size)
        df = next(df)
    else:
        df = pd.read_csv(input_fpath)

    vals = df.values
    print(vals.shape, vals[0, 0], vals[-1, 0])

    voltage = vals[:, 3]

    print(len(np.where(voltage > 0)[0]), len(np.where(voltage < 0)[0]))

    idx = np.where(voltage >= 0)[0]

    voltage = voltage[idx]
    current = vals[idx, 2]
    power = current * np.abs(voltage)
    windspeed = vals[idx, 4]
    ids = vals[idx, 0]
    dates = vals[idx, 1]

    t_start = time()

    last_id = ids[-1]
    str_last_date = pd.to_datetime(str(dates[-1])).strftime('%d/%m/%Y %H:%M:%S')
    print(str_last_date)

    values = []

    for wi, w in enumerate(np.arange(windspeed_min, windspeed_max, windspeed_step)):
        idx = np.where((windspeed >= w) & (windspeed <= w + windspeed_step))[0]
        n_power = len(idx)
        if n_power > 0:
            current_total = np.sum(current[idx])
            current_min = np.min(current[idx])
            current_max = np.max(current[idx])
            voltage_total = np.sum(voltage[idx])
            voltage_min = np.min(voltage[idx])
            voltage_max = np.max(voltage[idx])
            power_total = np.sum(power[idx])
            power_min = np.min(power[idx])
            power_max = np.max(power[idx])
        else:
            power_total = None
            power_min = None
            power_max = None
        values.append([
            last_id if wi == 0 else None,
            str_last_date if wi == 0 else None,
            #wi + 1,
            w,
            w + windspeed_step,
            n_power,
            current_total,
            current_min,
            current_max,
            voltage_total,
            voltage_min,
            voltage_max,
            power_total,
            power_min,
            power_max
        ])

    last_sample_id_col = id_col
    last_sample_timestamp_col = timestamp_col
    windspeed_interval_id = 'IntervalId'
    cols = [
        'WindSpeedLeft',
        'WindSpeedRight',
        'NumberOfSamples',
        'TotalCurrent',
        'MinimalCurrent',
        'MaximalCurrent',
        'TotalVoltage',
        'MinimalVoltage',
        'MaximalVoltage',
        'TotalPower',
        'MinimalPower',
        'MaximalPower',
    ]
    all_cols = [last_sample_id_col] + [windspeed_interval_id] + cols

    sqlalchemy_last_id_col = sqlalchemy.Column(last_sample_id_col, sqlalchemy.INT)
    sqlalchemy_last_timestamp_col = sqlalchemy.Column(last_sample_timestamp_col, sqlalchemy.String)
    sqlalchemy_primary_key_col = sqlalchemy.Column(windspeed_interval_id, sqlalchemy.INT, primary_key=True)
    sqlalchemy_cols = [sqlalchemy.Column(col, sqlalchemy.Float) for col in cols]

    params = urllib.parse.quote_plus(db_connection_str)
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    meta = sqlalchemy.MetaData(engine)
    table_pointer = sqlalchemy.Table(
        table,
        meta,
        sqlalchemy_last_id_col,
        sqlalchemy_last_timestamp_col,
        sqlalchemy_primary_key_col,
        *sqlalchemy_cols
    )
    meta.create_all(checkfirst=True)
    conn = engine.connect()

    query = f"insert into {table} ({','.join([last_sample_id_col, last_sample_timestamp_col] + cols)}) values ({','.join(['?' for _ in [last_sample_id_col, last_sample_timestamp_col] + cols])})"
    print(query)
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