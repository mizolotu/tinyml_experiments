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

table = 'WindsidePowerSummaryAllTime'

# database columns

id_col = 'ID'
timestamp_col = 'Date'
interval_id_col = 'IntervalId'
col_names = [
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

# auxiliary functions

def prepare_summary(cols_summary, rows_summary):

    timestamp_last = [row[cols_summary.index(timestamp_col)] for row in rows_summary if row[cols_summary.index(timestamp_col)] is not None][0]    
    ws_left = [float(row[cols_summary.index(col_names[0])]) for row in rows_summary]
    ws_right = [float(row[cols_summary.index(col_names[1])]) for row in rows_summary]
    powers_n = [int(row[cols_summary.index(col_names[2])]) for row in rows_summary]
    currents_total = [float(row[cols_summary.index(col_names[3])]) if row[cols_summary.index(col_names[3])] is not None else 0 for row in rows_summary]
    currents_min = [float(row[cols_summary.index(col_names[4])]) if row[cols_summary.index(col_names[4])] is not None else None for row in rows_summary]
    currents_max = [float(row[cols_summary.index(col_names[5])]) if row[cols_summary.index(col_names[5])] is not None else None for row in rows_summary]
    voltages_total = [float(row[cols_summary.index(col_names[6])]) if row[cols_summary.index(col_names[6])] is not None else 0 for row in rows_summary]
    voltages_min = [float(row[cols_summary.index(col_names[7])]) if row[cols_summary.index(col_names[7])] is not None else None for row in rows_summary]
    voltages_max = [float(row[cols_summary.index(col_names[8])]) if row[cols_summary.index(col_names[8])] is not None else None for row in rows_summary]
    powers_total = [float(row[cols_summary.index(col_names[9])]) if row[cols_summary.index(col_names[9])] is not None else 0 for row in rows_summary]
    powers_min = [float(row[cols_summary.index(col_names[10])]) if row[cols_summary.index(col_names[10])] is not None else None for row in rows_summary]
    powers_max = [float(row[cols_summary.index(col_names[11])]) if row[cols_summary.index(col_names[11])] is not None else None for row in rows_summary]

    lines = [
        f'Last updated: {timestamp_last}',
        '',
        'Wind speed, # samples, Avg current, Min current, Max current, Avg voltage, Min voltage, Max voltage, Avg power, Min power, Max power'
    ]

    for w_l, w_r, p_n, c_t, c_mn, c_mx, v_t, v_mn, v_mx, p_t, p_mn, p_mx in zip(ws_left, ws_right, powers_n, currents_total, currents_min, currents_max, voltages_total, voltages_min, voltages_max, powers_total, powers_min, powers_max):

        line = f'{w_l:.1f} - {w_r:.1f}, {p_n}, {c_t / p_n if p_n > 0 else None}, {c_mn if p_n > 0 else None}, {c_mx if p_n > 0 else None}, {v_t / p_n if p_n > 0 else None}, {v_mn if p_n > 0 else None}, {v_mx if p_n > 0 else None}, {p_t / p_n if p_n > 0 else None}, {p_mn if p_n > 0 else None}, {p_mx if p_n > 0 else None}'
        #logging.info(line)
        lines.append(line)
    
    return '\n'.join(lines)

def select_current_summary():
    query = f"select * from {table} order by {col_names[0]}"
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