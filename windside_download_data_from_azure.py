import pyodbc
import pandas as pd

from time import time
from config import *
from datetime import datetime
from db_config import password

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
##driver = '{ODBC Driver 17 for SQL Server}'
driver = '{SQL Server}'
username = 'jyusqlserver'
#password = ''
db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

def select_latest_n(db_connection_str, table, timestamp_col, n=1, ts=None):
    query = f"select top ({n}) * from {table} order by {timestamp_col} desc"
    rows = []
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [col[0] for col in cursor.description]
            assert timestamp_col in cols
            timestamp_col_idx = cols.index(timestamp_col)
            for row in cursor.fetchall():
                if ts is None or row[timestamp_col_idx] > ts:
                    rows.append(list(row))
    return cols, rows

def select_data_samples(tbl, timestamp_col, ts_last, n=100):
    cols, rows = select_latest_n(tbl, timestamp_col, ts=ts_last, n=n)
    assert timestamp_col in cols
    ts_idx = cols.index(timestamp_col)
    timestamps = [row.pop(ts_idx) for row in rows]
    cols.pop(ts_idx)
    return cols, rows, timestamps

def download_in_chunks(table, id_col, timestamp_col, chunk_size, n_chunks=None, last_timestamp=0):

    cols = None
    rows = []

    n = chunk_size
    i = 0

    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            while n == chunk_size:
                print(f'Downloading rows from {i * chunk_size} to {(i + 1) * chunk_size}')
                query = f'select * from {table} order by {id_col} desc offset {i * chunk_size} rows fetch next {chunk_size} rows only'
                tstart = time()
                cursor.execute(query)
                if cols is None:
                    cols = [col[0] for col in cursor.description]
                    timestamp_col_idx = cols.index(timestamp_col)
                new_rows = []
                for row in cursor.fetchall():
                    new_rows.append(list(row))
                n = len(new_rows)
                current_date = new_rows[n - 1][timestamp_col_idx]
                print('The last date in the chunk:', current_date)
                if type(current_date) == datetime:
                    current_timestamp = current_date.timestamp()
                elif type(current_date) == str:
                    current_timestamp = datetime.strptime(current_date, '%d/%m/%Y %H:%M:%S').timestamp()
                else:
                    raise NotImplemented
                rows.extend(new_rows)
                print(f'Downloaded: {n} rows in {time() - tstart:.2f} seconds')
                i += 1

                if current_timestamp < last_timestamp:
                    break
                elif n_chunks is not None and i >= n_chunks:
                    break

    return cols, rows

if __name__ == '__main__':

    # table and columns

    table = 'IoTdata2'
    id_col = 'ID'
    timestamp_col = 'date4'

    # output fpath

    dpath = osp.join(DATA_DIR, 'windside', 'raw')
    #dpath = osp.join(DATA_DIR, 'IoTdata')
    for d in [DATA_DIR, dpath]:
        if not osp.isdir(d):
            os.mkdir(d)

    fnames = [int(item.split('.csv')[0]) for item in os.listdir(dpath)]
    fname = str(max(fnames) + 1) if len(fnames) > 0 else '0'
    fname = ''.join(['0' for _ in range(2 - len(fname))]) + fname
    fname += '.csv'
    fpath = osp.join(dpath, fname)
    print('Current:', fpath)

    # get the latest date

    if len(fnames) > 0:
        fname_last = str(max(fnames))
        fname_last = ''.join(['0' for _ in range(2 - len(fname_last))]) + fname_last
        fname_last += '.csv'
        fpath_last = osp.join(dpath, fname_last)
        print('Last:', fpath_last)
        df_last = pd.read_csv(fpath_last, chunksize=1)
        df_last = next(df_last)
        last_timestamp = datetime.strptime(df_last[timestamp_col].values[0], '%d/%m/%Y %H:%M:%S').timestamp()
    else:
        last_timestamp = 0

    print('Last timestamp:', last_timestamp)

    if 0:
        #q = "select count(ID) FROM [dbo].[IoTdata2] where ID < 4777330;"
        #q = f"delete FROM {table} where ID < 4777330;"
        q = ";WITH CTE AS (SELECT TOP 200000 * FROM [dbo].[IoTdata2] ORDER BY ID ) DELETE FROM CTE"

        while True:

            try:
                with pyodbc.connect(db_connection_str) as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(q)
                        rows = [row for row in cursor.fetchall()]

                print(rows)
                break

            except Exception as e:
                print(e)
                sleep(3)

    # download all

    cols, rows = download_in_chunks(table, id_col, timestamp_col, chunk_size=100000, n_chunks=100, last_timestamp=last_timestamp)
    pd.DataFrame(rows, columns=cols).to_csv(fpath, index=False)