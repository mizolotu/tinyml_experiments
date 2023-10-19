import logging
import os, pyodbc, pytz
import numpy as np

from datetime import datetime

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
dst_table = 'WindsideHourData'

# database columns

id_col = 'ID'

src_timestamp_col = 'date4'
src_cols = [
    'value2',  # current
    'value1',  # voltage
    'value3',  # windspeed
]

dst_start_date_col = 'StartDate'
dst_start_timestamp_col = 'StartTimestamp'
dst_end_date_col = 'EndDate'
dst_end_timestamp_col = 'EndTimestamp'
dst_id_col = 'LastId'
dst_cols = [    
    'NumberOfSamples',

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

# tz

hlk_tz = pytz.timezone('Europe/Helsinki')

# auxiliary functions

def insert_new_values(cols_new, rows_new, ts_step=3600):

    id_new = np.array([row[cols_new.index(id_col)] for row in rows_new])
    timestamp_new = np.array([row[cols_new.index(src_timestamp_col)] for row in rows_new])    
    current_new = np.array([row[cols_new.index(src_cols[0])] for row in rows_new])
    voltage_new = np.array([row[cols_new.index(src_cols[1])] for row in rows_new])
    windspeed_new = np.array([row[cols_new.index(src_cols[2])] for row in rows_new])    
    
    idx = np.where(voltage_new > 0)[0][::-1]
    
    if len(idx) > 0: 
        
        id_new = id_new[idx]
        timestamp_new = timestamp_new[idx]
        current_new = current_new[idx]
        voltage_new = voltage_new[idx]
        windspeed_new = windspeed_new[idx]
        power_new = current_new * voltage_new
        unixtime_new = np.array([hlk_tz.localize(datetime.strptime(ts, '%d/%m/%Y %H:%M:%S')).timestamp() for ts in timestamp_new])
        #unixtime_new = np.array([datetime.strptime(ts, '%d/%m/%Y %H:%M:%S').timestamp() for ts in timestamp_new])

        timestamp_min = unixtime_new[0]
        timestamp_max = unixtime_new[-1]

        logging.info(timestamp_min)
        logging.info(timestamp_max)

        n_timestamps = int((timestamp_max - timestamp_min) / ts_step) + 1

        offset = 0

        for i in range(n_timestamps):

            s_ts = timestamp_min // ts_step * ts_step + i * ts_step
            s_ts_dt = datetime.fromtimestamp(s_ts).astimezone(hlk_tz).strftime('%d/%m/%Y %H:%M:%S')
            e_ts = s_ts + ts_step
            e_ts_dt = datetime.fromtimestamp(e_ts).astimezone(hlk_tz).strftime('%d/%m/%Y %H:%M:%S')

            idx = offset + np.where((unixtime_new[offset:offset+ts_step] >= s_ts) & (unixtime_new[offset:offset+ts_step] < e_ts))[0]
            n_power = len(idx)

            vals = {}
            if n_power > 0:

                vals[dst_start_date_col] = s_ts_dt    # start date
                vals[dst_start_timestamp_col] = s_ts  # start timestamp
                vals[dst_end_date_col] = e_ts_dt      # end date
                vals[dst_end_timestamp_col] = e_ts    # end timestamp
                
                vals[dst_id_col] = id_new[idx[-1]]    # last id

                vals[dst_cols[0]] = n_power

                vals[dst_cols[1]] = np.mean(current_new[idx])
                vals[dst_cols[2]] = np.min(current_new[idx])
                vals[dst_cols[3]] = np.max(current_new[idx])

                vals[dst_cols[4]] = np.mean(voltage_new[idx])
                vals[dst_cols[5]] = np.min(voltage_new[idx])
                vals[dst_cols[6]] = np.max(voltage_new[idx])

                vals[dst_cols[7]] = np.mean(power_new[idx])
                vals[dst_cols[8]] = np.min(power_new[idx])
                vals[dst_cols[9]] = np.max(power_new[idx])

                vals[dst_cols[10]] = np.mean(windspeed_new[idx])
                vals[dst_cols[11]] = np.min(windspeed_new[idx])
                vals[dst_cols[12]] = np.max(windspeed_new[idx])

                vals_u = [f"'{vals[key]}'" for key in vals.keys()]
                
                query = f"insert into {dst_table} values ({','.join(vals_u)});"

                logging.info(query)
                #with pyodbc.connect(db_connection_str) as conn:
                #    with conn.cursor() as cursor:
                #       cursor.execute(query)

                offset = idx[-1]
    
def select_latest_n_samples(n=100, id=None):
    query = f"select top ({n}) * from {src_table} order by {id_col} desc"
    #logging.info(query)
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

def select_last_id():
    query = f"select top (1) * from {dst_table} order by {id_col} desc"
    #logging.info(query)
    rows = []
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                rows.append(list(row))
    assert dst_id_col in cols
    id_idx = cols.index(dst_id_col)
    ids = [row.pop(id_idx) for row in rows]
    id_last = ids[0]
    return id_last