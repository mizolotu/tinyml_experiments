import pyodbc, os, urllib, sqlalchemy
import pandas as pd
import os.path as osp
import numpy as np

from time import time
from config import *
from db_config import password
from datetime import datetime

if __name__ == '__main__':

    # input fpath

    input_fpath = osp.join(DATA_DIR, 'windside', 'raw', '2.csv')

    # db access

    server = 'tcp:jyusqlserver.database.windows.net'
    database = 'IoTSQL'
    driver = '{SQL Server}'
    username = 'jyusqlserver'
    db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

    # table and columns

    table = 'WindSpeedPrediction'
    id_col = 'ID'
    timestamp_col = 'date4'

    # select data

    chunk_size = 3000
    start_date = '2022-10-01 00:00:00'
    n_samples = chunk_size
    date_col = 'date4'
    value_names = ['Current', 'WindSpeed', 'Voltage']

    date_parser = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

    if chunk_size is not None:
        df = pd.read_csv(input_fpath, parse_dates=[date_col], date_parser=date_parser, chunksize=chunk_size)
        df = next(df)
    else:
        df = pd.read_csv(input_fpath, parse_dates=[date_col], date_parser=date_parser)

    df = df.sort_values(by=[date_col])
    dates = df[date_col].values

    print(dates[0], dates[-1])

    start_date = pd.to_datetime(start_date, format='%Y-%m-%d %H:%M:%S')
    row_idx = np.where(dates >= start_date)[0][:n_samples]
    print(row_idx[0], row_idx[-1])

    str_dates = [pd.to_datetime(str(date)).strftime('%d/%m/%Y %H:%M:%S') for date in dates[row_idx]]

    header = [key for key in df.keys()]
    line1 = df.values[0, :]
    cols = [date_col]
    other_cols = []
    for name in value_names:
        assert name in line1
        idx = np.where(line1 == name)[0][0]
        cols.append(header[idx - 1])
        other_cols.append(sqlalchemy.Column(header[idx - 1], sqlalchemy.Float))
        cols.append(header[idx])
        other_cols.append(sqlalchemy.Column(header[idx], sqlalchemy.String))

    values = [[str_date, *row.tolist()] for str_date, row in zip(str_dates, df[cols[1:]].values[row_idx, :])]

    params = urllib.parse.quote_plus(db_connection_str)
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    meta = sqlalchemy.MetaData(engine)
    table_pointer = sqlalchemy.Table(
        table,
        meta,
        sqlalchemy.Column(id_col, sqlalchemy.Integer, primary_key=True),
        sqlalchemy.Column(date_col, sqlalchemy.String),
        *other_cols
    )
    meta.create_all(checkfirst=True)
    conn = engine.connect()

    query = f"insert into {table} ({','.join(cols)}) values ({','.join(['?' for _ in cols])})"
    print(query)
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, values)