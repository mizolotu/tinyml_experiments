import logging, pytz
import os, pyodbc, base64, io
import numpy as np

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
password = '#jyusql1'
db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

# database tables

lora_table = 'WindsideIoT'

# database columns

id_col = 'ID'

lora_date_col = 'date'

lora_cols = [
    'Current',
    'Voltage',
    'Power',
    'Windspeed',
]

# tz

hlk_tz = pytz.timezone('Europe/Helsinki')

# auxiliary functions

def generate_html(url, params):

    #logging.info(url)
    #logging.info(params)

    if 'code' in params.keys():
        code = params["code"]
    else:
        code = None

    #logging.info(code)

    sdate = None
    if 'sdate' in params.keys():
        try:
            sdate = params['sdate']            
        except Exception as e:
            logging.info(e)
            
    edate = None
    if 'edate' in params.keys():
        try:
            edate = params['edate']            
        except Exception as e:
            logging.info(e)
    
    cols, rows = _select_samples_between_lora(sdate=sdate, edate=edate)
    
    #logging.info('SELECTED!')
    #logging.info(rows[-1])

    labels = [
        'Virta, A',
        'Jännite, V',
        'Teho, W',
        'T. nopeus, m/s'
    ]

    colors = ['b', 'r', 'm', 'g']

    if len(rows) > 0:

        #logging.info('Preparing series...')
        dates, timestamps, currents, voltages, powers, windspeeds, sdate_, edate_ = prepare_series(cols, rows, sdate, edate)
        #logging.info('PREPARED!')
        
        #logging.info('Preparing hists...')
        hist_lines = prepare_hists(cols, rows)
        #logging.info('PREPARED!')

        data_to_plot = [currents, voltages, powers, windspeeds]

        _, axes = pp.subplots(4, 1, figsize=(25, 8))

        timestamps_u, timestamps_u_idx = np.unique(timestamps, return_index=True)
        n_u = len(timestamps_u)

        x = [i for i in range(n_u)]
        xticks = [i for i in range(0, n_u, np.maximum(1, n_u // 10))]
        xticklabels = [dates[timestamps_u_idx[i]] for i in xticks]
        
        for i, ax in enumerate(axes):

            #ax.bar(x, avg_currents, color='b', label='Average')
            ax.plot(x, data_to_plot[i]['avg'][timestamps_u_idx], colors[i], label='Average')
            ax.fill_between(x, data_to_plot[i]['min'][timestamps_u_idx], data_to_plot[i]['max'][timestamps_u_idx], facecolor=colors[i], alpha=0.3, label='Min/max')
            ax.set_xlim([x[0], x[-1]])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('Aikaleima')
            ax.set_ylabel(labels[i])
            ax.legend()
        
        buff = io.BytesIO()
        pp.subplots_adjust(hspace=0.5)
        pp.savefig(buff, format='svg', bbox_inches='tight', pad_inches=0)
        pp.close()

        img_byte_arr = buff.getvalue()
        img_str = base64.b64encode(img_byte_arr).decode()
        #logging.info(img_str)
        series_img_tag = f'<img src="data:image/svg+xml;base64,{img_str}" style="width:100%"/>'

        # table1

        dates = np.array([dates[i].split(' ')[0].split('/') for i in timestamps_u_idx], dtype=int)

        table1_lines = []

        table_headers = [
            'Vuosi',
            'Kuukausi',
            'Päivä'
        ]

        table1_lines = []

        for k in range(3):

            table_header = table_headers[k:k+1] + labels

            periods = dates[:, 2-k:]

            periods_u = np.unique(periods, axis=0)

            if k == 0:
                idx = np.arange(len(periods_u))
            elif k == 1:
                idx = np.argsort(periods_u[:, 1])
            elif k == 2:
                idx = np.lexsort((periods_u[:, 1], periods_u[:, 2]))
            periods_u = periods_u[idx, :]

            #logging.info(periods_u)

            table_line = ['<p><table style="width:100%;" allign="left"><tr>']
            for item in table_header:
                table_line.append(f'<th align="right">{item}</th>')
            table_line.append('</tr>')

            table1_lines.append(''.join(table_line))
            
            for i, period in enumerate(periods_u):

                if k == 0:
                    idx = np.where(periods == period)[0]
                elif k == 1:
                    idx = np.where((periods[:, 0] == period[0]) & (periods[:, 1] == period[1]))[0]
                elif k == 2:
                    idx = np.where((periods[:, 0] == period[0]) & (periods[:, 1] == period[1]) & (periods[:, 2] == period[2]))[0]
                
                values = [
                    '/'.join([str(item) for item in period]),
                    f"{np.mean(currents['avg'][idx]):.4f} / {np.max(currents['max'][idx]):.4f}",
                    f"{np.mean(voltages['avg'][idx]):.4f} / {np.max(voltages['max'][idx]):.4f}",
                    f"{np.mean(powers['avg'][idx]):.4f} / {np.max(powers['max'][idx]):.4f}",
                    f"{np.mean(windspeeds['avg'][idx]):.4f} / {np.max(windspeeds['max'][idx]):.4f}"
                ]

                table_line = ['<tr>']
                for j, item in enumerate(values):
                    if j == 0:
                        table_line.append(f'<td align="right">{item}</td>')
                    else:
                        item_formatted = item # f'{item:.6f}' if item is not None else None
                        table_line.append(f'<td align="right">{item_formatted}</td>')
                table_line.append('</tr>')        
                table1_lines.append(''.join(table_line))
                
            table1_lines.append('</table></p><hr>')

        table1_lines = ''.join(table1_lines)

        # hists

        windspeeds, currents, voltages, powers = {}, {}, {}, {}

        for key in ['avg', 'min', 'max']:
            windspeeds[key] = [line[0] for line in hist_lines[key] if None not in line]
            currents[key] = [line[3] for line in hist_lines[key] if None not in line]
            voltages[key] = [line[4] for line in hist_lines[key] if None not in line]
            powers[key] = [line[5] for line in hist_lines[key] if None not in line]
                
        _, axes = pp.subplots(1, 3, figsize=(25, 4))

        data_to_plot = [currents, voltages, powers]
        key_to_plot = 'max'

        xticks = [i for i in range(1 + int(windspeeds[key_to_plot][-1]))]

        for i, ax in enumerate(axes):
            ax.plot(windspeeds[key_to_plot], data_to_plot[i][key_to_plot], f'{colors[i]}o-', label=key_to_plot.capitalize())
            ax.set_xticks(xticks)
            ax.set_xlabel('T. nopeus, m/s')
            ax.set_ylabel(labels[i])
            ax.set_xlim([windspeeds[key_to_plot][0], windspeeds[key_to_plot][-1]])
            ax.legend()

        buff = io.BytesIO()
        pp.savefig(buff, format='svg', bbox_inches='tight', pad_inches=0)
        pp.close()

        img_byte_arr = buff.getvalue()
        img_str = base64.b64encode(img_byte_arr).decode()
        #logging.info(img_str)
        hist_img_tag = f'<img src="data:image/svg+xml;base64,{img_str}" style="width:100%"/>'

        # table2

        table2_lines = []

        table_header = [
            labels[-1],
            'Näytteiden lukumäärä',
            #'Avg current',
            #'Min current',
            #'Max current',
            f'{labels[0]} ({key_to_plot})',
            #'Avg voltage',
            #'Min voltage',
            #'Max voltage',
            f'{labels[1]} ({key_to_plot})',
            #'Avg power',
            #'Min power',
            #'Max power'
            f'{labels[2]} ({key_to_plot})'
        ]    

        table_line = ['<table style="width:100%"><tr>']
        for item in table_header:
            table_line.append(f'<th align="right">{item}</th>')
        table_line.append('</tr>')
        table2_lines.append(''.join(table_line))

        for i, summary_line in enumerate(hist_lines[key_to_plot]):
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
            table2_lines.append(''.join(table_line))
        
        table2_lines.append('</table>')
        table2_lines = ''.join(table2_lines)

    else:
        sdate_ = sdate
        edate_ = edate
        series_img_tag = None
        hist_img_tag = None

    # form tag

    form_tag = [
        f'<form action="{url}">',
        #'<label for="sensor">Sensor:</label><br>'
        #f'<label class="radio-inline"><input type="radio" id="lte" name="sensor" value="lte" {"checked" if sensor == "lte" else ""}>LTE</input></label>',
        #f'<label class="radio-inline"><input type="radio" id="lora" name="sensor" value="lora" {"checked" if sensor == "lora" else ""}>LoRa</input</label><br>',
        f'<label for="sdate">Start date:</label><br><input type="text" id="sdate" name="sdate" value="{sdate_}"><br>',
        f'<label for="edate">End date:</label><br><input type="text" id="edate" name="edate" value="{edate_}"><br>',
        f'<input type="hidden" id="code" name="code" value="{code}"/>',
        '<br><input type="submit" value="Submit"></form>'
    ]
    form_tag = ''.join(form_tag)

    # interval 
            
    interval_line = f'Päivämäärä: {sdate_} - {edate_}'

    html = [f'<html><head></head><body>{form_tag}<p>{interval_line}</p>']

    html.append('<div style="width: 100%;"><div style="float:left; width: 60%;">')
    
    if series_img_tag is not None:
        html.append(f'<p>Timestamp vs. current, voltage, power and wind speed:</p>')
        html.append(f'<p style="width:100%">{series_img_tag}</p>')

    if hist_img_tag is not None:
        html.append(f'<p>Wind speed vs. current, voltage and power for {key_to_plot.upper()} values:</p>')
        html.append(f'<p style="width:100%">{hist_img_tag}</p>')
        html.append(f'<p>{table2_lines}</p>')

    html.append('</div><div style="float: right; width: 39%; allign: left">')

    if series_img_tag is not None:
        html.append('<p><small>*All the metric values are printed in format: average / maximum.</small></p>')
        html.append(table1_lines)

    html.append('</div></div>')
        
    html.append('</body></html>')

    return ''.join(html)

def prepare_series(cols, rows, sdate, edate, ts_step=60*10):

    start_dates = [row[cols.index(lora_date_col)] for row in rows]

    unixtimes = np.array([sd.timestamp() for sd in start_dates])
    currents_ = np.array([row[cols.index(lora_cols[0])] for row in rows])        
    voltages_ = np.array([row[cols.index(lora_cols[1])] for row in rows])
    powers_ = np.array([row[cols.index(lora_cols[2])] for row in rows])        
    windspeeds_ = np.array([row[cols.index(lora_cols[3])] for row in rows])

    sdate_ = start_dates[0].strftime('%d/%m/%Y %H:%M:%S') if sdate is None else sdate 
    timestamp_min = datetime.strptime(sdate_, '%d/%m/%Y %H:%M:%S').timestamp()
    edate_ = start_dates[-1].strftime('%d/%m/%Y %H:%M:%S') if edate is None else edate
    timestamp_max = datetime.strptime(edate_, '%d/%m/%Y %H:%M:%S').timestamp()

    n_timestamps = int((timestamp_max - timestamp_min) / ts_step) + 1

    offset = 0
        
    start_dates = []
    start_timestamps = []

    currents = {'avg': [], 'min': [], 'max': []}
    voltages = {'avg': [], 'min': [], 'max': []}
    powers = {'avg': [], 'min': [], 'max': []}
    windspeeds = {'avg': [], 'min': [], 'max': []}
        
    for i in range(n_timestamps):

        s_ts = timestamp_min // ts_step * ts_step + i * ts_step
        s_ts_dt = datetime.fromtimestamp(s_ts).strftime('%d/%m/%Y %H:%M:%S')
        e_ts = s_ts + ts_step
        e_ts_dt = datetime.fromtimestamp(e_ts).strftime('%d/%m/%Y %H:%M:%S')
            
        idx = offset + np.where((unixtimes[offset:offset+ts_step] >= s_ts) & (unixtimes[offset:offset+ts_step] < e_ts))[0]
        n_power = len(idx)

        #logging.info(n_power)

        if n_power > 0:

            start_dates.append(s_ts_dt)
            start_timestamps.append(s_ts)

            currents['avg'].append(np.mean(currents_[idx]))                
            currents['min'].append(np.min(currents_[idx]))
            currents['max'].append(np.max(currents_[idx]))

            voltages['avg'].append(np.mean(voltages_[idx]))
            voltages['min'].append(np.min(voltages_[idx]))
            voltages['max'].append(np.max(voltages_[idx]))

            powers['avg'].append(np.mean(powers_[idx]))                
            powers['min'].append(np.min(powers_[idx]))
            powers['max'].append(np.max(powers_[idx]))

            windspeeds['avg'].append(np.mean(windspeeds_[idx]))                
            windspeeds['min'].append(np.min(windspeeds_[idx]))
            windspeeds['max'].append(np.max(windspeeds_[idx]))
                
            offset = idx[-1]

        else:                
            #logging.info(offset)
            idx_tmp = np.where(unixtimes[offset:] <= e_ts)[0]
            if len(idx_tmp) > 0:
                offset = offset + idx_tmp[-1]

    start_dates = np.array(start_dates)
    for key in ['avg', 'min', 'max']:
        currents[key] = np.array(currents[key])
        voltages[key] = np.array(voltages[key])
        powers[key] = np.array(powers[key])
        windspeeds[key] = np.array(windspeeds[key])

    return start_dates, start_timestamps, currents, voltages, powers, windspeeds, sdate_, edate_


def prepare_hists(cols, rows, ws_step=0.1):

    hist_lines = {
        'avg': [], 
        'min': [], 
        'max': []
    }

    currents = {
        'avg': np.array([row[cols.index(lora_cols[0])] for row in rows]),
        'min': np.array([row[cols.index(lora_cols[0])] for row in rows]),
        'max': np.array([row[cols.index(lora_cols[0])] for row in rows])
    }
    
    voltages = {
        'avg': np.array([row[cols.index(lora_cols[1])] for row in rows]),
        'min': np.array([row[cols.index(lora_cols[1])] for row in rows]),
        'max': np.array([row[cols.index(lora_cols[1])] for row in rows])
    }

    windspeeds = {
        'avg': np.array([row[cols.index(lora_cols[3])] for row in rows]),
        'min': np.array([row[cols.index(lora_cols[3])] for row in rows]),
        'max': np.array([row[cols.index(lora_cols[3])] for row in rows])
    }
   
    # wind speed bins

    for key in hist_lines.keys(): 

        ws_min = float(min(windspeeds[key]))
        ws_max = float(max(windspeeds[key]))
        n_ws = int((ws_max - ws_min) / ws_step) + 2

        for i in range(n_ws):

            ws = ws_min // ws_step * ws_step + i * ws_step

            idx = np.where((windspeeds[key] >= ws) & (windspeeds[key] <= ws + ws_step) & (voltages[key] >= 0))[0]

            currents_i = currents[key][idx]
            voltages_i = voltages[key][idx]
            powers = currents_i * voltages_i
                            
            power_n = len(powers)

            power_avg = sum(powers) / power_n if power_n > 0 else None
            current_avg = sum(currents_i) / power_n if power_n > 0 else None
            voltage_avg = sum(voltages_i) / power_n if power_n > 0 else None

            line = [ws, ws + ws_step, power_n, current_avg, voltage_avg, power_avg]
            #logging.info(line)
            hist_lines[key].append(line)

    return hist_lines

def _select_samples_between_lora(sdate=None, edate=None):

    rows = []

    query = [f"select * from {lora_table}"]
    if sdate is not None or edate is not None:
        query.append(' where ')
        if sdate is not None:
            sdate_formatted = datetime.strptime(sdate, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            query.append(f"{lora_date_col} >= '{sdate_formatted}'")
            if edate is not None:
                edate_formatted = datetime.strptime(edate, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                query.append(f" AND {lora_date_col} <= '{edate_formatted}'")
        else:
            edate_formatted = datetime.strptime(edate, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            query.append(f"{lora_date_col} <= '{edate_formatted}'")
    query.append(f' order by {lora_date_col}')

    query = ''.join(query)

    #logging.info(query)
    
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                rows.append(list(row))

    return cols, rows