import logging
import os, pyodbc, base64, io

from matplotlib import pyplot as pp
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

data_table = 'IoTdata2'
summary_table = 'WindsidePowerSummaryAllTime'

# database columns

id_col = 'ID'

data_timestamp_col = 'date4'

data_cols = [
    'value2',  # current
    'value1',  # voltage
    'value3',  # windspeed
]

summary_timestamp_col = 'Date'
interval_id_col = 'IntervalId'

summary_cols = [
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

def prepare_summary(lookback=None, timestamp_first='11/05/2022 00:00:00', ws_step=0.1, ts_step=60):

    summary_lines = []
    bars = []    

    #logging.info(lookback)

    if lookback is None:

        _, cols_summary, rows_summary = _get_all_time_summary()

        timestamp_last = [row[cols_summary.index(summary_timestamp_col)] for row in rows_summary if row[cols_summary.index(summary_timestamp_col)] is not None][0]    
        ws_left = [float(row[cols_summary.index(summary_cols[0])]) for row in rows_summary]
        ws_right = [float(row[cols_summary.index(summary_cols[1])]) for row in rows_summary]
        powers_n = [int(row[cols_summary.index(summary_cols[2])]) for row in rows_summary]
        currents_total = [float(row[cols_summary.index(summary_cols[3])]) if row[cols_summary.index(summary_cols[3])] is not None else 0 for row in rows_summary]
        currents_min = [float(row[cols_summary.index(summary_cols[4])]) if row[cols_summary.index(summary_cols[4])] is not None else None for row in rows_summary]
        currents_max = [float(row[cols_summary.index(summary_cols[5])]) if row[cols_summary.index(summary_cols[5])] is not None else None for row in rows_summary]
        voltages_total = [float(row[cols_summary.index(summary_cols[6])]) if row[cols_summary.index(summary_cols[6])] is not None else 0 for row in rows_summary]
        voltages_min = [float(row[cols_summary.index(summary_cols[7])]) if row[cols_summary.index(summary_cols[7])] is not None else None for row in rows_summary]
        voltages_max = [float(row[cols_summary.index(summary_cols[8])]) if row[cols_summary.index(summary_cols[8])] is not None else None for row in rows_summary]
        powers_total = [float(row[cols_summary.index(summary_cols[9])]) if row[cols_summary.index(summary_cols[9])] is not None else 0 for row in rows_summary]
        powers_min = [float(row[cols_summary.index(summary_cols[10])]) if row[cols_summary.index(summary_cols[10])] is not None else None for row in rows_summary]
        powers_max = [float(row[cols_summary.index(summary_cols[11])]) if row[cols_summary.index(summary_cols[11])] is not None else None for row in rows_summary]

        ws_max = [i for i, pn in enumerate(powers_n) if pn > 0]
        ws_max = ws_right[ws_max[-1]]

        for w_l, w_r, p_n, c_t, c_mn, c_mx, v_t, v_mn, v_mx, p_t, p_mn, p_mx in zip(ws_left, ws_right, powers_n, currents_total, currents_min, currents_max, voltages_total, voltages_min, voltages_max, powers_total, powers_min, powers_max):

            line = [w_l, w_r, p_n, c_t / p_n if p_n > 0 else None, c_mn if p_n > 0 else None, c_mx if p_n > 0 else None, v_t / p_n if p_n > 0 else None, v_mn if p_n > 0 else None, v_mx if p_n > 0 else None, p_t / p_n if p_n > 0 else None, p_mn if p_n > 0 else None, p_mx if p_n > 0 else None]
            #logging.info(line)
            if w_l < ws_max:
                summary_lines.append(line)

    else:

        cols, rows = _select_latest_n_samples(n=int(lookback if lookback >= 0 else 0) + 1, chunk_size=10000)

        timestamp_new = [row[cols.index(data_timestamp_col)] for row in rows]
        current_new = [row[cols.index(data_cols[0])] for row in rows]
        voltage_new = [row[cols.index(data_cols[1])] for row in rows]
        windspeed_new = [row[cols.index(data_cols[2])] for row in rows]    

        unixtime_new = [datetime.strptime(ts, '%d/%m/%Y %H:%M:%S').timestamp() for ts in timestamp_new]
        ut_last = unixtime_new[0]
        
        if lookback > 0:
            idx = [i for i, ut in enumerate(unixtime_new) if ut > ut_last - lookback]
        else:
            idx = [i for i, ut in enumerate(unixtime_new) if ut == ut_last]
        
        currents = [current_new[i] for i in idx]
        voltages = [voltage_new[i] for i in idx]
        windspeeds = [windspeed_new[i] for i in idx]
        unixtimes = [unixtime_new[i] for i in idx]
        
        timestamp_first = timestamp_new[idx[-1]]
        timestamp_last = timestamp_new[0]

        # time bins

        ts_min = float(min(unixtime_new))
        ts_max = float(max(unixtime_new))
        n_ts = int((ts_max - ts_min) / ts_step) + 1

        for i in range(n_ts):

            ts = ts_min // ts_step * ts_step + i * ts_step
            ts_dt = datetime.fromtimestamp(ts)

            currents_i = [float(c) for t, c, v in zip(unixtimes, currents, voltages) if t >= ts and t <= ts + ts_step and v >= 0]
            voltages_i = [float(v) for t, v in zip(unixtimes, voltages) if t >= ts and t <= ts + ts_step and v >= 0]
            windspeeds_i = [float(w) for t, w, v in zip(unixtimes, windspeeds, voltages) if t >= ts and t <= ts + ts_step and v >= 0]
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

            windspeed_avg = sum(windspeeds_i) / power_n if power_n > 0 else None
            windspeed_min = min(windspeeds_i) if power_n > 0 else None
            windspeed_max = max(windspeeds_i) if power_n > 0 else None
                        
            line = [ts_dt, current_avg, current_min, current_max, voltage_avg, voltage_min, voltage_max, power_avg, power_min, power_max, windspeed_avg, windspeed_min, windspeed_max]
            #logging.info(line)
            bars.append(line)
            
        # wind speed bins

        ws_min = float(min(windspeeds))
        ws_max = float(max(windspeeds))
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
                        
            line = [ws, ws + ws_step, power_n, current_avg, current_min, current_max, voltage_avg, voltage_min, voltage_max, power_avg, power_min, power_max]
            #logging.info(line)
            summary_lines.append(line)

    return timestamp_first, timestamp_last, summary_lines, bars

def generate_html(timestamp_first, timestamp_last, summary_lines, bars):

    if timestamp_first is not None and timestamp_last is not None:

        # interval
            
        interval_line = f'Interval: {timestamp_first} - {timestamp_last}'        

        # header
        
        table_lines = []

        table_header = [
            'Wind speed bin',
            'Number of samples',
            'Avg current',
            'Min current',
            'Max current',
            'Avg voltage',
            'Min voltage',
            'Max voltage',
            'Avg power',
            'Min power',
            'Max power'
        ]    

        table_line = ['<table style="width:100%"><tr>']
        for item in table_header:
            table_line.append(f'<th align="right">{item}</th>')
        table_line.append('</tr>')
        table_lines.append(''.join(table_line))

        # values
        
        for i, summary_line in enumerate(summary_lines):
            table_line = ['<tr>']
            for j, item in enumerate(summary_line):
                if j == 0:
                    table_line.append(f'<td align="right">{item:.1f} - {summary_line[1]:.1f}</td>')
                elif j == 2:
                    table_line.append(f'<td align="right">{item}</td>')
                elif j >= 3:
                    item_formatted = f'{item:.6f}' if item is not None else None
                    table_line.append(f'<td align="right">{item_formatted}</td>')
            table_line.append('</tr>')        
            table_lines.append(''.join(table_line))
        
        table_lines.append('</table>')
        table_lines = ''.join(table_lines)

        if len(bars) > 0:
            
            timestamps = [line[0] for line in bars if None not in line]
            avg_currents = [line[1] for line in bars if None not in line]
            min_currents = [line[2] for line in bars if None not in line]
            max_currents = [line[3] for line in bars if None not in line]
            avg_voltages = [line[4] for line in bars if None not in line]
            min_voltages = [line[5] for line in bars if None not in line]
            max_voltages = [line[6] for line in bars if None not in line]
            avg_powers = [line[7] for line in bars if None not in line]
            min_powers = [line[8] for line in bars if None not in line]
            max_powers = [line[9] for line in bars if None not in line]
            avg_windspeeds = [line[10] for line in bars if None not in line]
            min_windspeeds = [line[11] for line in bars if None not in line]
            max_windspeeds = [line[12] for line in bars if None not in line]

            _, ax = pp.subplots(2, 2, figsize=(25, 4))            

            x = [i for i in range(len(avg_currents))]
            xticks = [i for i in range(0, len(avg_currents), len(avg_currents) // 5)]
            xticklabels = [timestamps[i] for i in xticks]

            ax[0, 0].bar(x, avg_currents, color='b', label='Average')
            ax[0, 0].set_xlim([x[0], x[-1]])
            ax[0, 0].set_xticks(xticks)
            ax[0, 0].set_xticklabels(xticklabels)
            ax[0, 0].set_xlabel('Timestamp')
            ax[0, 0].set_ylabel('Current, A')
            ax[0, 0].legend()

            ax[0, 1].bar(x, avg_voltages, color='r', label='Average')
            ax[0, 1].set_xlim([x[0], x[-1]])            
            ax[0, 1].set_xticks(xticks)
            ax[0, 1].set_xticklabels(xticklabels)
            ax[0, 1].set_xlabel('Timestamp')
            ax[0, 1].set_ylabel('Voltage, V')
            ax[0, 1].legend()

            ax[1, 0].bar(x, avg_windspeeds, color='g', label='Average')
            ax[1, 0].set_xlim([x[0], x[-1]])
            ax[1, 0].set_xticks(xticks)
            ax[1, 0].set_xticklabels(xticklabels)
            ax[1, 0].set_xlabel('Timestamp')
            ax[1, 0].set_ylabel('Windspeed, m/s')
            ax[1, 0].legend()

            ax[1, 1].bar(x, avg_powers, color='m', label='Average')
            ax[1, 1].set_xlim([x[0], x[-1]])
            ax[1, 1].set_xticks(xticks)
            ax[1, 1].set_xticklabels(xticklabels)
            ax[1, 1].set_xlabel('Timestamp')
            ax[1, 1].set_ylabel('Power, W')
            ax[1, 1].legend()

            buff = io.BytesIO()
            pp.savefig(buff, format='svg', bbox_inches='tight', pad_inches=0)

            img_byte_arr = buff.getvalue()
            img_str = base64.b64encode(img_byte_arr).decode()
            #logging.info(img_str)
            bar_tag = f'<img src="data:image/svg+xml;base64,{img_str}" style="width:100%"/>'
        
        else:
            bar_tag = None

        windspeeds = [line[0] for line in summary_lines if None not in line]
        
        avg_currents = [line[3] for line in summary_lines if None not in line]
        min_currents = [line[4] for line in summary_lines if None not in line]
        max_currents = [line[5] for line in summary_lines if None not in line]

        avg_voltages = [line[6] for line in summary_lines if None not in line]
        min_voltages = [line[7] for line in summary_lines if None not in line]
        max_voltages = [line[8] for line in summary_lines if None not in line]

        avg_powers = [line[9] for line in summary_lines if None not in line]
        min_powers = [line[10] for line in summary_lines if None not in line]
        max_powers = [line[11] for line in summary_lines if None not in line]
        
        xticks = [i for i in range(1 + int(windspeeds[-1]))]

        _, ax = pp.subplots(1, 3, figsize=(25, 4))

        ax[0].plot(windspeeds, avg_currents, 'b-', label='Average')
        ax[0].fill_between(windspeeds, min_currents, max_currents, facecolor='b', alpha=0.3, label='Min/max')
        ax[0].set_xlim([windspeeds[0], windspeeds[-1]])
        ax[0].set_xticks(xticks)
        ax[0].set_xlabel('Wind speed, m/s')
        ax[0].set_ylabel('Current, A')
        ax[0].legend()

        ax[1].plot(windspeeds, avg_voltages, 'r-', label='Average')
        ax[1].fill_between(windspeeds, min_voltages, max_voltages, facecolor='r', alpha=0.3, label='Min/max')
        ax[1].set_xlim([windspeeds[0], windspeeds[-1]])
        ax[1].set_xticks(xticks)
        ax[1].set_xlabel('Wind speed, m/s')
        ax[1].set_ylabel('Voltage, V')
        ax[1].legend()

        ax[2].plot(windspeeds, avg_powers, 'm-', label='Average')
        ax[2].fill_between(windspeeds, min_powers, max_powers, facecolor='m', alpha=0.3, label='Min/max')
        ax[2].set_xlim([windspeeds[0], windspeeds[-1]])
        ax[2].set_xticks(xticks)
        ax[2].set_xlabel('Wind speed, m/s')
        ax[2].set_ylabel('Power, W')
        ax[2].legend()

        buff = io.BytesIO()
        pp.savefig(buff, format='svg', bbox_inches='tight', pad_inches=0)

        img_byte_arr = buff.getvalue()
        img_str = base64.b64encode(img_byte_arr).decode()
        #logging.info(img_str)
        img_tag = f'<img src="data:image/svg+xml;base64,{img_str}" style="width:100%"/>'
        
        if bar_tag is not None:
            html = f'<html><head></head><body><p>{interval_line}</p><p style="width:100%">{bar_tag}</p><p style="width:100%">{img_tag}</p><p>{table_lines}</p></body></html>'
        else:
            html = f'<html><head></head><body><p>{interval_line}</p><p style="width:100%">{img_tag}</p><p>{table_lines}</p></body></html>'

    return html


def _get_all_time_summary():
    query = f"select * from {summary_table} order by {summary_cols[0]}"
    #logging.info(query)
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
        
def _select_latest_n_samples(n=100, chunk_size=10000):

    rows = []
    n_chunks = n // chunk_size 

    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:

            for i in range(n_chunks):

                query = f"select * from {data_table} order by {id_col} desc offset {i * chunk_size} rows fetch next {chunk_size} rows only"
    
                cursor.execute(query)
                cols = [col[0] for col in cursor.description]
                for row in cursor.fetchall():
                    rows.append(list(row))

            if n_chunks * chunk_size < n:

                query = f"select * from {data_table} order by {id_col} desc offset {n_chunks * chunk_size} rows fetch next {n - n_chunks * chunk_size} rows only"
    
                cursor.execute(query)
                cols = [col[0] for col in cursor.description]
                for row in cursor.fetchall():
                    rows.append(list(row))

    return cols, rows