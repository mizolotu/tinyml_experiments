import logging
import os, pyodbc

# path

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

# sql connection

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
driver = '{ODBC Driver 17 for SQL Server}'
username = 'jyusqlserver'
password = ''
db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

# database tables

src_table = 'IoTdata2'
dst_table = 'WindsidePowerSummaryAllTime'

# database columns

id_col = 'ID'

src_timestamp_col = 'date4'
src_cols = [
    'value2',  # current
    'value1',  # voltage
    'value3',  # windspeed
]

dst_timestamp_col = 'Date'
dst_id_col = 'IntervalId'
dst_cols = [
    'WindSpeedLeft',
    'WindSpeedRight',
    'NumberOfSamples',
    'TotalPower',
    'MinimalPower',
    'MaximalPower',
]

# auxiliary functions

def update_summary(cols_summary, rows_summary, cols_new, rows_new):

    values = []

    id_new = [row[cols_new.index(id_col)] for row in rows_new]
    timestamp_new = [row[cols_new.index(src_timestamp_col)] for row in rows_new]
    current_new = [row[cols_new.index(src_cols[0])] for row in rows_new]
    voltage_new = [row[cols_new.index(src_cols[1])] for row in rows_new]
    windspeed_new = [row[cols_new.index(src_cols[2])] for row in rows_new]    
    id_last = id_new[0]
    timestamp_last = timestamp_new[0]
    logging.info(id_last)
    logging.info(timestamp_last)

    for row_i, row in enumerate(rows_summary):

        w_interval_id = row_i + 1            
        w_left = float(row[cols_summary.index(dst_cols[0])])
        w_right = float(row[cols_summary.index(dst_cols[1])])
        power_n = int(row[cols_summary.index(dst_cols[2])])
        power_total = row[cols_summary.index(dst_cols[3])]
        power_total = float(power_total) if power_total is not None else 0
        power_min = row[cols_summary.index(dst_cols[4])]
        power_min = float(power_min) if power_min is not None else None
        power_max = row[cols_summary.index(dst_cols[5])]
        power_max = float(power_max) if power_max is not None else None
    
        powers = [float(c) * float(v) for w, c, v in zip(windspeed_new, current_new, voltage_new) if float(w) >= w_left and float(w) <= w_right and v >= 0]
        n_power = len(powers)

        vals = {}
        if n_power > 0:            
            vals[dst_cols[2]] = power_n + n_power
            vals[dst_cols[3]] = power_total + sum(powers)
            vals[dst_cols[4]] = min(power_min, min(powers)) if power_min is not None else min(powers)
            vals[dst_cols[5]] = max(power_max, max(powers)) if power_max is not None else max(powers)

            vals_u = [f'{key} = {vals[key]}' for key in vals.keys()]

            logging.info(f'{row_i}: {w_left} - {w_right}, powers: {powers}')
            query = f"update {dst_table} set {','.join(vals_u)} where {dst_id_col} = {w_interval_id};"
            logging.info(query)
            with pyodbc.connect(db_connection_str) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
            
        values.append([
            id_last if row_i == 0 else None,
            w_left,
            w_right,
            power_n,
            power_total,
            power_min,
            power_max
        ])

    query = f"update {dst_table} set {id_col} = {id_last} where {id_col} is not null"
    logging.info(query)
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)

    query = f"update {dst_table} set {dst_timestamp_col} = '{timestamp_last}' where {dst_timestamp_col} is not null"
    logging.info(query)
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)

def select_latest_n_samples(n=100, id=None):
    query = f"select top ({n}) * from {src_table} order by {id_col} desc"
    logging.info(query)
    rows = []
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [col[0] for col in cursor.description]
            assert id_col in cols
            id_col_idx = cols.index(id_col)
            for row in cursor.fetchall():
                if id is None or row[id_col_idx] > id:
                    rows.append(list(row))
    return cols, rows

def select_current_summary():
    query = f"select * from {dst_table} order by {dst_cols[0]}"
    logging.info(query)
    rows = []
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                rows.append(list(row))
    assert id_col in cols
    id_idx = cols.index(id_col)
    ids = [row.pop(id_idx) for row in rows]
    cols.pop(id_idx)
    id_last = ids[0]
    return id_last, cols, rows