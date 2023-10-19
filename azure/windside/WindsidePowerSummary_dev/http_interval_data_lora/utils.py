import logging
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
password = ''
db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

# database tables

summary_table = 'WindsideIoT'

# database columns

id_col = 'ID'

date_col = 'date'

data_cols = [
    'Current',
    'Voltage',
    'Power',
    'Windspeed',
]

# auxiliary functions

def generate_html(url, params, ts_step=60*10, sdate_min='01/06/2023 00:00:00'):

    #logging.info(url)
    #logging.info(params)

    if 'code' in params.keys():
        code = params["code"]
    else:
        code = None

    #logging.info(code)

    sdate = sdate_min
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

    cols, rows = _select_samples_between(sdate=sdate, edate=edate)

    if len(rows) > 0:

        labels = [
            'Virta, A',
            'Jännite, V',
            'Teho, W',
            'T. nopeus, m/s'
        ]

        colors = ['b', 'r', 'm', 'g']

        summary_lines = prepare_summary(cols, rows)

        start_dates = [row[cols.index(date_col)] for row in rows]

        unixtimes = np.array([sd.timestamp() for sd in start_dates])
        currents = np.array([row[cols.index(data_cols[0])] for row in rows])        
        voltages = np.array([row[cols.index(data_cols[1])] for row in rows])
        powers = np.array([row[cols.index(data_cols[2])] for row in rows])        
        windspeeds = np.array([row[cols.index(data_cols[3])] for row in rows])


        sdate_ = start_dates[0].strftime('%d/%m/%Y %H:%M:%S') if sdate is None else sdate 
        timestamp_min = datetime.strptime(sdate_, '%d/%m/%Y %H:%M:%S').timestamp()
        edate_ = start_dates[-1].strftime('%d/%m/%Y %H:%M:%S') if edate is None else edate
        timestamp_max = datetime.strptime(edate_, '%d/%m/%Y %H:%M:%S').timestamp()

        n_timestamps = int((timestamp_max - timestamp_min) / ts_step) + 1

        offset = 0
        
        avg_currents, min_currents, max_currents = [], [], []
        avg_voltages, min_voltages, max_voltages = [], [], []
        avg_powers, min_powers, max_powers = [], [], []
        avg_windspeeds, min_windspeeds, max_windspeeds = [], [], []
        dates = []

        for i in range(n_timestamps):

            s_ts = timestamp_min // ts_step * ts_step + i * ts_step
            s_ts_dt = datetime.fromtimestamp(s_ts).strftime('%d/%m/%Y %H:%M:%S')
            e_ts = s_ts + ts_step
            e_ts_dt = datetime.fromtimestamp(e_ts).strftime('%d/%m/%Y %H:%M:%S')

            idx = offset + np.where((unixtimes[offset:offset+ts_step] >= s_ts) & (unixtimes[offset:offset+ts_step] < e_ts))[0]
            n_power = len(idx)

            #logging.info(n_power)

            if n_power > 0:

                dates.append(s_ts_dt)

                avg_currents.append(np.mean(currents[idx]))
                min_currents.append(np.min(currents[idx]))
                max_currents.append(np.max(currents[idx]))

                avg_voltages.append(np.mean(voltages[idx]))
                min_voltages.append(np.min(voltages[idx]))
                max_voltages.append(np.max(voltages[idx]))

                avg_powers.append(np.mean(powers[idx]))
                min_powers.append(np.min(powers[idx]))
                max_powers.append(np.max(powers[idx]))

                avg_windspeeds.append(np.mean(windspeeds[idx]))
                min_windspeeds.append(np.min(windspeeds[idx]))
                max_windspeeds.append(np.max(windspeeds[idx]))

                offset = idx[-1]

            else:                
                #logging.info(offset)
                idx_tmp = np.where(unixtimes[offset:] <= e_ts)[0]
                if len(idx_tmp) > 0:
                    offset = offset + idx_tmp[-1]
        
        # series

        #logging.info(len(dates))

        if len(dates) > 0:

            data_to_plot = [
                [avg_currents, min_currents, max_currents],
                [avg_voltages, min_voltages, max_voltages],
                [avg_powers, min_powers, max_powers],
                [avg_windspeeds, min_windspeeds, max_windspeeds],
            ]

            _, axes = pp.subplots(4, 1, figsize=(25, 8))            

            #x = [i for i in range(len(avg_currents))]
            #xticks = [i for i in range(0, len(avg_currents), len(avg_currents) // 10)]
            #xticklabels = [dates[i] for i in xticks]

            n_u = len(dates)
            x = [i for i in range(n_u)]
            xticks = [i for i in range(0, n_u, np.maximum(1, n_u // 10))]
            xticklabels = [dates[i] for i in xticks]

            for i, ax in enumerate(axes):

                ax.plot(x, data_to_plot[i][0], colors[i], label='Average')
                ax.fill_between(x, data_to_plot[i][1], data_to_plot[i][2], facecolor=colors[i], alpha=0.3, label='Min/max')
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

            #logging.info(start_dates)

            #dates = np.array([item.split(' ')[0].split('/') for item in start_dates], dtype=int)
            dates = np.array([[item.day, item.month, item.year] for item in start_dates], dtype=int)

            table1_lines = []

            table_headers = [
                'Vuosi',
                'Kuukausi',
                'Päivä'
            ]

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
                        np.mean(currents[idx]),
                        np.mean(voltages[idx]),
                        np.mean(powers[idx]),
                        np.mean(windspeeds[idx])
                    ]

                    table_line = ['<tr>']
                    for j, item in enumerate(values):
                        if j == 0:
                            table_line.append(f'<td align="right">{item}</td>')
                        else:
                            item_formatted = f'{item:.6f}' if item is not None else None
                            table_line.append(f'<td align="right">{item_formatted}</td>')
                    table_line.append('</tr>')        
                    table1_lines.append(''.join(table_line))
            
                table1_lines.append('</table></p><hr>')

            table1_lines = ''.join(table1_lines)

        # hists

        windspeeds = [line[0] for line in summary_lines if None not in line]
        
        avg_currents = [line[3] for line in summary_lines if None not in line]
        #avg_currents = [line[3] for line in summary_lines if None not in line]
        #min_currents = [line[4] for line in summary_lines if None not in line]
        #max_currents = [line[5] for line in summary_lines if None not in line]

        avg_voltages = [line[4] for line in summary_lines if None not in line]
        #avg_voltages = [line[6] for line in summary_lines if None not in line]
        #min_voltages = [line[7] for line in summary_lines if None not in line]
        #max_voltages = [line[8] for line in summary_lines if None not in line]

        avg_powers = [line[5] for line in summary_lines if None not in line]
        #avg_powers = [line[9] for line in summary_lines if None not in line]
        #min_powers = [line[10] for line in summary_lines if None not in line]
        #max_powers = [line[11] for line in summary_lines if None not in line]
        
        xticks = [i for i in range(1 + int(windspeeds[-1]))]

        _, axes = pp.subplots(1, 3, figsize=(25, 4))

        data_to_plot = [
            [avg_currents],
            [avg_voltages],
            [avg_powers],
        ]

        for i, ax in enumerate(axes):

            ax.plot(windspeeds, data_to_plot[i][0], f'{colors[i]}o-', label='Average')
            #ax.fill_between(windspeeds, data_to_plot[i][1], data_to_plot[i][2], facecolor=colors[i], alpha=0.3, label='Min/max')            
            ax.set_xticks(xticks)
            ax.set_xlabel('T. nopeus, m/s')
            ax.set_ylabel(labels[i])
            ax.set_xlim([windspeeds[0], windspeeds[-1]])
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
            labels[0],
            #'Avg voltage',
            #'Min voltage',
            #'Max voltage',
            labels[1],
            #'Avg power',
            #'Min power',
            #'Max power'
            labels[2]
        ]    

        table_line = ['<table style="width:100%"><tr>']
        for item in table_header:
            table_line.append(f'<th align="right">{item}</th>')
        table_line.append('</tr>')
        table2_lines.append(''.join(table_line))

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
        html.append(f'<p style="width:100%">{series_img_tag}</p>')

    if hist_img_tag is not None:
        html.append(f'<p>Wind speed vs. current, voltage and power for {key_to_plot} values:</p>')
        html.append(f'<p style="width:100%">{hist_img_tag}</p>')
        html.append(f'<p>{table2_lines}</p>')
    
    html.append('</div><div style="float: right; width: 39%; allign: left">')

    if series_img_tag is not None:
        html.append('<p><small>*All the metric values are printed in format: average / maximum.</small></p>')
        html.append(table1_lines)

    html.append('</div></div>')
        
    html.append('</body></html>')

    return ''.join(html)

def prepare_summary(cols, rows, ws_step=0.1):

    summary_lines = []    
    
    avg_currents = np.array([row[cols.index(data_cols[0])] for row in rows])
    avg_voltages = np.array([row[cols.index(data_cols[1])] for row in rows])
    avg_windspeeds = np.array([row[cols.index(data_cols[3])] for row in rows])
       
    # wind speed bins

    ws_min = float(min(avg_windspeeds))
    ws_max = float(max(avg_windspeeds))
    n_ws = int((ws_max - ws_min) / ws_step) + 1

    for i in range(n_ws):

        ws = ws_min // ws_step * ws_step + i * ws_step

        idx = np.where((avg_windspeeds >= ws) & (avg_windspeeds <= ws + ws_step) & (avg_voltages >= 0))[0]

        currents_i = avg_currents[idx]
        voltages_i = avg_voltages[idx]
        powers = currents_i * voltages_i
                            
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
                        
        #line = [ws, ws + ws_step, power_n, current_avg, current_min, current_max, voltage_avg, voltage_min, voltage_max, power_avg, power_min, power_max]
        line = [ws, ws + ws_step, power_n, current_avg, voltage_avg, power_avg]
        #logging.info(line)
        summary_lines.append(line)

    return summary_lines

def _select_samples_between(sdate=None, edate=None):

    rows = []

    query = [f"select * from {summary_table}"]
    if sdate is not None or edate is not None:
        query.append(' where ')
        if sdate is not None:
            sdate_formatted = datetime.strptime(sdate, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            query.append(f"{date_col} >= '{sdate_formatted}'")
            if edate is not None:
                edate_formatted = datetime.strptime(edate, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                query.append(f" AND {date_col} <= '{edate_formatted}'")
        else:
            edate_formatted = datetime.strptime(edate, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            query.append(f"{date_col} <= '{edate_formatted}'")
    query.append(f' order by {date_col}')

    query = ''.join(query)

    #logging.info(query)
    
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                rows.append(list(row))

    return cols, rows