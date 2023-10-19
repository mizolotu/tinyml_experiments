import logging
import os, pyodbc

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

table = 'IoTdata2'

# database columns

id_col = 'ID'

timestamp_col = 'date4'
cols = [
    'value2',  # current
    'value1',  # voltage
    'value3',  # windspeed
]

# auxiliary functions

def generate_summary(cols_new, rows_new, lookback=60):

    timestamp_new = [row[cols_new.index(timestamp_col)] for row in rows_new]
    current_new = [row[cols_new.index(cols[0])] for row in rows_new]
    voltage_new = [row[cols_new.index(cols[1])] for row in rows_new]
    windspeed_new = [row[cols_new.index(cols[2])] for row in rows_new]    
    unixtime_new = [datetime.strptime(ts, '%d/%m/%Y %H:%M:%S').timestamp() for ts in timestamp_new]

    ut_last = unixtime_new[0]
    idx = [i for i, ut in enumerate(unixtime_new) if ut > ut_last - lookback]
    
    currents = [current_new[i] for i in idx]
    voltages = [voltage_new[i] for i in idx]
    windspeeds = [windspeed_new[i] for i in idx]

    timestamp_last = timestamp_new[0]
    
    lines = [
        f'Interval: {timestamp_new[idx[-1]]} - {timestamp_last}',
        '',
        'Wind speed, # samples, Avg current, Min current, Max current, Avg voltage, Min voltage, Max voltage, Avg power, Min power, Max power'
    ]

    # bins

    ws_min = float(min(windspeeds))
    ws_max = float(max(windspeeds))
    ws_step = 0.1
    n_ws = int((ws_max - ws_min) / ws_step) + 1

    for i in range(n_ws):

        ws = ws_min // ws_step * ws_step + i * ws_step

        currents_i = [float(c) for w, c, v in zip(windspeeds, currents, voltages) if w >= ws and w <= ws + ws_step and v >= 0]
        voltages_i = [float(v) for w, v in zip(windspeeds, voltages) if w >= ws and w <= ws + ws_step and v >= 0]
        powers = [c * v for c, v in zip(currents_i, voltages_i)]
                        
        power_n = len(powers)

        power_avg = sum(powers) / power_n if power_n > 0 else None
        power_min = min(powers) if power_n > 0 else None
        power_max = max(powers) if power_n > 0 else None

        current_avg = sum(currents_i) / power_n if power_n > 0 else None
        current_min = min(currents_i) if power_n > 0 else None
        current_max = max(currents_i) if power_n > 0 else None

        voltage_avg = sum(voltages_i) / power_n if power_n > 0 else None
        voltage_min = min(voltages_i) if power_n > 0 else None
        voltage_max = max(voltages_i) if power_n > 0 else None
        
        line = f'{ws:.1f} - {ws + ws_step:.1f}, {power_n}, {current_avg}, {current_min}, {current_max}, {voltage_avg}, {voltage_min}, {voltage_max}, {power_avg}, {power_min}, {power_max}'
        
        lines.append(line)
    
    return '\n'.join(lines)

        
def select_latest_n_samples(n=100, chunk_size=10000):

    rows = []
    n_chunks = n // chunk_size 
    
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:

            for i in range(n_chunks):

                query = f"select * from {table} order by {id_col} desc offset {i * chunk_size} rows fetch next {chunk_size} rows only"
    
                cursor.execute(query)
                cols = [col[0] for col in cursor.description]
                for row in cursor.fetchall():
                    rows.append(list(row))

            if n_chunks * chunk_size < n:

                query = f"select * from {table} order by {id_col} desc offset {n_chunks * chunk_size} rows fetch next {n - n_chunks * chunk_size} rows only"
    
                cursor.execute(query)
                cols = [col[0] for col in cursor.description]
                for row in cursor.fetchall():
                    rows.append(list(row))

    return cols, rows