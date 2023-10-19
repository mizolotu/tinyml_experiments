import logging, serial, pyodbc
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px

from dash import html, dcc, callback, Output, Input, register_page
from config import *
from datetime import datetime
from time import time, sleep, mktime
from threading import Thread
from collections import deque

from db_config import password

register_page(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
driver = '{ODBC Driver 17 for SQL Server}'
username = 'jyusqlserver'
db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'
table = 'IoTdata2'
id_col = 'ID'
date_col = 'date4'
voltage = 'Voltage'

c_min, c_max = 0, 2
v_min, v_max = 24, 26
w_min, w_max = 0, 4
time_zone = 2

def select_latest_n(db_connection_str, table, id_col, n=1):
    query = f"select top ({n}) * from {table} order by {id_col} desc"
    rows = []
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                rows.append(list(row))
    return cols, rows

def select_n_after_offset(db_connection_str, table, id_col, offset=0, n=1):
	query = f'select * from {table} order by {id_col} offset {offset} rows fetch next {n} rows only'
	rows = []
	with pyodbc.connect(db_connection_str) as conn:
		with conn.cursor() as cursor:
			cursor.execute(query)
			cols = [col[0] for col in cursor.description]
			for row in cursor.fetchall():
				rows.append(list(row))
	return cols, rows

def extract_data(cols, rows, date_col, value_names=['Current', 'Voltage', 'WindSpeed']):
	line1 = rows[0]
	features = []
	for name in value_names:
		assert name in line1
		idx = line1.index(name)
		features.append([float(row[idx - 1]) for row in rows])
	date_idx = cols.index(date_col)
	#D = [np.datetime64(datetime.strptime(row[date_idx], '%d/%m/%Y %H:%M:%S')) for row in rows][::-1]
	D = [mktime(datetime.strptime(row[date_idx], '%d/%m/%Y %H:%M:%S').timetuple()) for row in rows][::-1]
	XY = np.array(features).T[::-1]
	X = XY[:, :-1]
	Y = XY[:, -1]

	return D, X, Y

select_some = select_latest_n

serial_port ='/dev/ttyACM1'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate)

feature_extractor = 'raw'
data_fpath = osp.join(DATA_DIR, 'windside', 'features.csv')

#X, Y, D = create_dataset(data_fpath, feature_extractor=feature_extractor, window_size=1, chunk_size=10000, feature_cols=['Current', 'Voltage'])
#_, _, _, X_inf, Y_inf, D_inf, _ = load_and_split_dataset((X, Y, D))
#print('Shapes: ', X_inf.shape, Y_inf.shape)

small_delay = 1e-3
series_len = 60
window_step = 10
plot_len = 60 + series_len

sql_q = deque(maxlen=plot_len)
data_q = deque(maxlen=plot_len)
pred_q = deque(maxlen=plot_len)

print(f'Getting first data from {table}...')
cols, rows = select_some(db_connection_str, table, id_col, n=series_len + window_step)
D, X, Y = extract_data(cols, rows, date_col=date_col)

n_samples = len(Y)
for i in range(n_samples):
	sql_q.append((D[i], float(X[i, 0]), float(X[i, 1]), float(Y[i])))
print(f'Got {n_samples} samples!')

#last_unix = mktime(datetime.strptime(sql_q[-1][0], '%d/%m/%Y %H:%M:%S').timetuple())
last_unix = sql_q[-1][0]
t_now = time()
sql_delay = np.maximum(t_now - last_unix, 0.0)

def get_sql(dst_q):
	print('sql_delay:', sql_delay)
	sleep(window_step)
	while True:
		t_now = time()
		#last_unix = mktime(datetime.strptime(sql_q[-1][0], '%d/%m/%Y %H:%M:%S').timetuple())
		last_unix = sql_q[-1][0]
		if t_now - last_unix > sql_delay:
			print(f'Getting new data from {table}...')
			cols, rows = select_some(db_connection_str, table, id_col, n=int(window_step + np.ceil(sql_delay)))
			D, X, Y = extract_data(cols, rows, date_col=date_col)
			n_samples = len(Y)
			count = 0
			for i in range(n_samples):
				#t = mktime(datetime.strptime(D[i], '%d/%m/%Y %H:%M:%S').timetuple())
				t = D[i]
				if t >= last_unix:
					dst_q.append((D[i], float(X[i, 0]), float(X[i, 1]), float(Y[i])))
					count += 1
			print(f'Got {count} new samples!')
			if (time() - t_now) < window_step:
				sleep(window_step - time() + t_now)
		else:
			sleep(small_delay)

def send_vector(port, x):
	#print('wrote', x)
	port.write(str.encode(f'{x}\n'))

def write_serial(port, src_q, dst_q, ctrl_q):
	sql_data = [src_q.popleft() for _ in range(series_len)]
	data = np.vstack(sql_data)
	#unix_times = [mktime(datetime.strptime(date, '%d/%m/%Y %H:%M:%S').timetuple()) for date in data[:, 0]]
	unix_times = data[:, 0]

	# send first 60 currents

	for i in range(series_len):
		c = data[i, 1]
		v = data[i, 2]
		w = data[i, 3]
		send_vector(port, c)
		dst_q.append((D[i], c, v, w))

	last_unix = unix_times[-1]

	# start sending one-by-one after the last timestamp

	while True:
		if len(dst_q) <= len(ctrl_q):
			t_start = time()
			data = np.vstack([x for x in src_q])
			#unix_times = np.array([mktime(datetime.strptime(date, '%d/%m/%Y %H:%M:%S').timetuple()) for date in data[:, 0]])
			unix_times = data[:, 0]
			idx = np.where(unix_times > last_unix)[0]
			if len(idx) > 0:
				i = idx[0]
				if len(idx) > 1:
					i_next = idx[1]
					delay = unix_times[i_next] - unix_times[i]
				else:
					delay = 1.0
				t = data[i, 0]
				c = data[i, 1]
				v = data[i, 2]
				w = data[i, 3]
				#print('Appending to the data q:', t, c, v, w)
				last_unix = unix_times[i]
				send_vector(port, c)
				dst_q.append((t, c, v, w))

				t_now = time()
				if t_now - t_start < delay:
					#print('sleep', delay - t_now + t_start)
					sleep(delay - t_now + t_start)
			else:
				#print('IDX:', idx, last_unix)
				pass

		else:
			#print(len(dst_q), len(ctrl_q))
			sleep(small_delay)

	try:
		port.close()
	except:
		pass

def receive_vector(ser, start_marker=60, end_marker=62):
	msg = ''
	x = 'z'
	while ord(x) != start_marker:
		x = ser.read()
	while ord(x) != end_marker:
		if ord(x) != start_marker:
			msg = f'{msg}{x.decode("utf-8")}'
		x = ser.read()

	#print('read', msg)

	result_dict = {}
	if '=' in msg:
		try:
			spl = msg.split(';')
			for part in spl:
				part_spl = part.split('=')
				key = part_spl[0]
				value = [float(item) for item in part_spl[1].split(',')]
				result_dict[key] = value
		except Exception as e:
			print(e)

	elif 'Important: ' in msg:
		result_dict['m'] = f"{datetime.now().strftime('%d.%m.%y %H:%M:%S.%f')}: {msg.split('Important: ')[1].capitalize()}"

	return result_dict, msg

def read_serial(port, dst_q, ctrl_q):
	while(True):
		if len(dst_q) < len(ctrl_q) or len(dst_q) == plot_len:
			try:
				x_dict, msg = receive_vector(port)
				if x_dict is not None and type(x_dict) == dict:
					for key in x_dict.keys():
						#print(key, x_dict[key])
						if key == 'wind_speed':
							x = x_dict[key][0]
							dst_q.append(x)
							#print('Appended to pred q:', x)
			except Exception as e:
				#print(e)
				pass
		else:
			#print(len(dst_q), len())
			sleep(small_delay)

get_sql_thr = Thread(target=get_sql, args=(sql_q,), daemon=False)
get_sql_thr.start()

write_serial_thr = Thread(target=write_serial, args=(ser, sql_q, data_q, pred_q), daemon=False)
write_serial_thr.start()

read_serial_thr = Thread(target=read_serial, args=(ser, pred_q, data_q), daemon=False)
read_serial_thr.start()

layout = html.Div(style={'height': '100%', 'width': '100%', 'top': 0, 'z-index': 1, 'overflow-x': 'hidden', 'padding-top': '0px', 'position': 'fixed'}, children=[

	dcc.Interval(id='interval', n_intervals=0, interval=1000),

	dcc.Graph(
		id='current_plot',
		figure={},
		style={'height': '33%', 'width': '100%'}
	),

	dcc.Graph(
		id='voltage_plot',
		figure={},
		style={'height': '33%', 'width': '100%'}
	),

	dcc.Graph(
		id='windspeed_plot',
		figure={},
		style={'height': '33%', 'width': '100%'}
	)

])

@callback(
	Output("current_plot", "figure"),
	Output("voltage_plot", "figure"),
    Output("windspeed_plot", "figure"),
    Input("interval", "n_intervals"),
	prevent_initial_call=True
)
def update_progress(n_intervals):

	data = np.vstack([x for x in data_q])[series_len:, :]
	#print('data.shape:', data.shape)

	predictions = np.vstack([x for x in pred_q])[series_len:, :]
	#print('predictions.shape:', predictions.shape)

	n_samples = data.shape[0]
	index = np.arange(n_samples).reshape(-1, 1)

	df = pd.DataFrame(np.hstack([index, data, predictions]), columns=['Index', 'Date', 'Current', 'Voltage', 'Real wind speed', 'Predicted wind speed'])

	#df = pd.DataFrame(data[data_idx, :], columns=['Index', 'Real', 'Predicted'])

	tick_vals, tick_texts  = [], []
	for i in range(np.minimum(data.shape[0], plot_len - series_len)):
		t = df['Date'].values[i]
		date = datetime.utcfromtimestamp(t + time_zone * 3600).strftime('%Y-%m-%d %H:%M:%S')
		if t % 60 == 0 and date not in tick_texts:
			tick_vals.append(i)
			tick_texts.append(date)

	current_fig = px.scatter(df, x='Index', y=['Current'], color_discrete_sequence=['green'], template='plotly_white')
	current_fig.update_layout(xaxis={'tickmode': 'array', 'tickvals': tick_vals, 'ticktext': tick_texts})
	current_fig.update_layout(xaxis_title='Date', yaxis_title='Current')
	current_fig.update_traces(mode='lines+markers')
	current_fig.update_xaxes(tickangle=0)
	current_fig.update_layout(margin={'l': 20, 'b': 20, 'r': 20, 't': 20}, legend={'yanchor': "top", 'y': 0.99, 'xanchor': "right", 'x':0.99}, legend_title_text='Metrics')
	current_fig.update_xaxes(range=[0, plot_len - series_len])
	#current_fig.update_yaxes(range=[np.min(X[:, 0]), np.max(X[:, 0])])
	current_fig.update_yaxes(range=[c_min, c_max])

	voltage_fig = px.scatter(df, x='Index', y=['Voltage'], color_discrete_sequence=['orange'], template='plotly_white')
	voltage_fig.update_layout(xaxis={'tickmode': 'array', 'tickvals': tick_vals, 'ticktext': tick_texts})
	voltage_fig.update_layout(xaxis_title='Date', yaxis_title='Voltage')
	voltage_fig.update_traces(mode='lines+markers')
	voltage_fig.update_xaxes(tickangle=0)
	voltage_fig.update_layout(margin={'l': 20, 'b': 20, 'r': 20, 't': 20}, legend={'yanchor': "top", 'y': 0.99, 'xanchor': "right", 'x':0.99}, legend_title_text='Metrics')
	voltage_fig.update_xaxes(range=[0, plot_len - series_len])
	#voltage_fig.update_yaxes(range=[np.min(X[:, 1]), np.max(X[:, 1])])
	voltage_fig.update_yaxes(range=[v_min, v_max])

	windspeed_fig = px.scatter(df, x='Index', y=['Real wind speed', 'Predicted wind speed'], template='plotly_white')
	windspeed_fig.update_layout(xaxis={'tickmode': 'array', 'tickvals': tick_vals, 'ticktext': tick_texts})
	windspeed_fig.update_layout(xaxis_title='Date', yaxis_title='Wind speed')
	windspeed_fig.update_traces(mode='lines+markers')
	windspeed_fig.update_xaxes(tickangle=0)
	windspeed_fig.update_layout(margin={'l': 20, 'b': 20, 'r': 20, 't': 20}, legend={'yanchor': "top", 'y': 0.99, 'xanchor': "right", 'x':0.99}, legend_title_text='Metrics')
	windspeed_fig.update_xaxes(range=[0, plot_len - series_len])
	#windspeed_fig.update_yaxes(range=[np.min(Y), np.max(Y)])
	windspeed_fig.update_yaxes(range=[w_min, w_max])

	return current_fig, voltage_fig, windspeed_fig