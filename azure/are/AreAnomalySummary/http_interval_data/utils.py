import logging
import os, pyodbc, io, base64

import numpy as np
from datetime import datetime
from time import time
from matplotlib import pyplot as pp

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

table = 'AreIoT'

# database columns

id_col = 'ID'
date_col = 'date'

n_models = 3
score_cols = [f'AvgScore{x + 1}' for x in range(n_models)] + [f'MaxScore{x + 1}' for x in range(n_models)] 

score_thr = 1.0

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
    
    cols, rows = _select_samples_between(sdate=sdate, edate=edate)
    
    #logging.info(rows[-1])

    score_labels = [
        'Deep SVDD',
        'Scalable KM++',
        'CluStream'
    ]

    avg_score_labels = ['Avg score']

    anomaly_counts_labels = ['Num. of anomalies']

    thr_label = 'Score threshold'
    
    thr_color = 'k'
    score_colors = ['b', 'r', 'm']

    keys_to_plots = ['avg', 'max', 'anomalies']
    n_plots = len(keys_to_plots)
    
    if len(rows) > 0:

        #logging.info('Preparing series...')
        dates, timestamps, scores, anomaly_counts, list_of_anomalies, sdate_, edate_ = _prepare_series(cols, rows, sdate, edate)
        #logging.info('PREPARED!')
                
        _, axes = pp.subplots(n_plots, 1, figsize=(25, 8))

        timestamps_u, timestamps_u_idx = np.unique(timestamps, return_index=True)
        print([dates[i] for i in np.arange(len(timestamps)) if i not in timestamps_u_idx])
        n_u = len(timestamps_u)

        x = np.array([i for i in range(n_u)])

        xticks = [i for i in range(0, n_u, np.maximum(1, n_u // 8))]
        xticklabels = [dates[timestamps_u_idx[i]] for i in xticks]
        
        for i, ax in enumerate(axes):
            
            if i < n_plots - 1:
                for j in range(n_models):
                    ax.plot(x, scores[j][keys_to_plots[i]][timestamps_u_idx], score_colors[j], label=score_labels[j])
                
                if i == 0:
                    ax.plot(x, [score_thr for _ in x], f'{thr_color}--', label=thr_label)
                    
                ax.set_ylabel(f'{keys_to_plots[i].capitalize()} scores')

            else:
                #ax.plot(x, anomaly_counts, thr_color, label=f'{keys_to_plots[i].capitalize()} counts')
                ax.plot(x, [0 for _ in x], thr_color)
                idx_a = np.where(anomaly_counts > 0)[0]
                if len(idx_a) > 0:
                    ax.stem(x[idx_a], anomaly_counts[idx_a], thr_color, basefmt=thr_color, label=f'{keys_to_plots[i].capitalize()}')
                ax.set_ylabel(f'{keys_to_plots[i].capitalize()}')
                        
            ax.set_xlim([x[0], x[-1]])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('Aikaleima')            
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

        for k in range(2, 3):

            table_header = table_headers[k:k+1] + score_labels + anomaly_counts_labels

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

            table_line = ['<hr><p><table style="width:100%;" allign="left"><tr>']
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
                
                values = ['/'.join([str(item) for item in period])]
                for j in range(n_models):
                    values.append(f"{np.mean(scores[j]['avg'][idx]):.4f} / {np.max(scores[j]['max'][idx]):.4f}")

                values.append(f"{np.sum(anomaly_counts[idx])}")
                    
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

        # table 2

        table2_lines = []
        
        table_header = ['Timestamp'] + score_labels + avg_score_labels

        table_line = ['<hr><p><table style="width:100%"><tr>']
        for item in table_header:
            table_line.append(f'<th align="right">{item}</th>')
        table_line.append('</tr>')
        table2_lines.append(''.join(table_line))

        for anomaly in list_of_anomalies:

            table_line = ['<tr>']

            for i, value in enumerate(anomaly):                
                
                if i == 0:
                    table_line.append(f"<td align='right'>{value.strftime('%d/%m/%Y %H:%M:%S')}</td>")
                else:
                    table_line.append(f"<td align='right'>{f'{value:.4f}'}</td>")
                    
            table_line.append('</tr>')        
            
            table2_lines.append(''.join(table_line))
                
        table2_lines.append('</table></p><hr>')
        table2_lines = ''.join(table2_lines)

    else:
        sdate_ = sdate
        edate_ = edate
        series_img_tag = None

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

    html.append('<div style="width: 100%;"><div style="float:left; width: 65%;">')
    
    if series_img_tag is not None:
        html.append(f'<p>Timestamp vs. Anomaly scores:</p>')
        html.append(f'<p style="width:100%">{series_img_tag}</p>')

        #html.append(f'<p>Timestamps of the anomalies detected:</p>')
        #html.append(f'<p>{table2_lines}</p>')

        html.append('<p>Day summary (score values are printed in format: average / maximum):<p>')
        html.append(table1_lines)

    html.append('</div><div style="float: right; width: 35%; allign: left">')

    if series_img_tag is not None:
        #html.append('<p>Day summary (score values are printed in format: average / maximum):<p>')
        #html.append(table1_lines)
        html.append(f'<p>Anomalies detected:</p>')
        html.append(f'<p>{table2_lines}</p>')


    html.append('</div></div>')
        
    html.append('</body></html>')

    return ''.join(html)


def _prepare_series_tw(cols, rows, sdate, edate, ts_step=60*5):

    start_dates_ = np.array([row[cols.index(date_col)] for row in rows])

    unixtimes = np.array([sd.timestamp() for sd in start_dates_])
    
    avg_scores = []    
    max_scores = []

    for i in range(n_models):
        avg_scores.append(np.array([row[cols.index(score_cols[i])] for row in rows]))
        max_scores.append(np.array([row[cols.index(score_cols[n_models + i])] for row in rows]))
    
    sdate_ = start_dates_[0].strftime('%d/%m/%Y %H:%M:%S') if sdate is None else sdate 
    timestamp_min = datetime.strptime(sdate_, '%d/%m/%Y %H:%M:%S').timestamp()
    edate_ = start_dates_[-1].strftime('%d/%m/%Y %H:%M:%S') if edate is None else edate
    timestamp_max = datetime.strptime(edate_, '%d/%m/%Y %H:%M:%S').timestamp()

    n_timestamps = int((timestamp_max - timestamp_min) / ts_step) + 1

    offset = 0
        
    start_dates = []
    start_timestamps = []

    score_dicts = []
    for i in range(n_models):
        score_dicts.append({'avg': [], 'max': []})
    
    list_of_anomalies = []
    anomaly_counts = []
       
    for i in range(n_timestamps):

        s_ts = timestamp_min // ts_step * ts_step + i * ts_step
        s_ts_dt = datetime.fromtimestamp(s_ts).strftime('%d/%m/%Y %H:%M:%S')
        e_ts = s_ts + ts_step
            
        idx = offset + np.where((unixtimes[offset:offset+ts_step] >= s_ts) & (unixtimes[offset:offset+ts_step] < e_ts))[0]
        n_samples = len(idx)

        #logging.info(f'{i}/{n_timestamps}: {n_samples}')

        if n_samples > 0:

            start_dates.append(s_ts_dt)
            start_timestamps.append(s_ts)
            start_dates_idx = start_dates_[idx]

            avg_scores_ = np.zeros((len(idx), n_models))
            for j in range(n_models):
                score_dicts[j]['avg'].append(np.mean(avg_scores[j][idx]))
                score_dicts[j]['max'].append(np.max(max_scores[j][idx]))
                avg_scores_[:, j] = avg_scores[j][idx]            
                        
            idx_a = np.where(np.mean(avg_scores_, 1) > score_thr)[0]
            
            anomaly_counts.append(len(idx_a))
            
            if len(idx_a) > 0:                
                list_of_anomalies.extend([(start_dates_idx[i_a], *avg_scores_[i_a, :], np.mean(avg_scores_[i_a, :])) for i_a in idx_a])
            
            offset = idx[-1]

        else:                
            
            #logging.info(offset)
            
            idx_tmp = np.where(unixtimes[offset:] <= e_ts)[0]
            if len(idx_tmp) > 0:
                offset = offset + idx_tmp[-1]

    start_dates = np.array(start_dates)
    
    for key in ['avg', 'max']:
        for i in range(n_models):
            score_dicts[i][key] = np.array(score_dicts[i][key])
    
    anomaly_counts = np.array(anomaly_counts)

    return start_dates, start_timestamps, score_dicts, anomaly_counts, list_of_anomalies, sdate_, edate_

def _prepare_series(cols, rows, sdate, edate):

    start_dates_ = np.array([row[cols.index(date_col)] for row in rows])
    start_timestamps = np.array([sd.timestamp() for sd in start_dates_])
    
    sdate_ = start_dates_[0].strftime('%d/%m/%Y %H:%M:%S') if sdate is None else sdate 
    edate_ = start_dates_[-1].strftime('%d/%m/%Y %H:%M:%S') if edate is None else edate

    start_dates = np.array([item.strftime('%d/%m/%Y %H:%M:%S') for item in start_dates_])
    
    score_dicts = []
    for i in range(n_models):
        score_dicts.append({
            'avg': np.array([row[cols.index(score_cols[i])] for row in rows]), 
            'max': np.array([row[cols.index(score_cols[n_models + i])] for row in rows])
        })

    avg_scores_ = np.vstack([score_dict['avg'] for score_dict in score_dicts])
    
    anomaly_counts = np.array([1 if x > score_thr else 0 for x in np.mean(avg_scores_, 0)])
    idx_a = np.where(anomaly_counts > 0)[0]
    
    list_of_anomalies = [(start_dates_[i], *avg_scores_[:, i], np.mean(avg_scores_[:, i])) for i in idx_a]
       
    return start_dates, start_timestamps, score_dicts, anomaly_counts, list_of_anomalies, sdate_, edate_

def _select_samples_between(sdate=None, edate=None):

    rows = []

    query = [f"select * from {table}"]
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