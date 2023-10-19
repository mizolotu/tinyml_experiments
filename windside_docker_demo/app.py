import dash_bootstrap_components as dbc
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import pyodbc

from dash import Dash, html, dcc, callback, Output, Input, callback_context
from collections import deque

from db_config import password

series_n = 60
window_step = 10
interval = 1
data_q_len = series_n + window_step - 1
plot_len = 100

data_q = deque(maxlen=data_q_len)
plot_q = deque(maxlen=plot_len)

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
#driver = '{SQL Server}'
driver = '{ODBC Driver 17 for SQL Server}'
username = 'jyusqlserver'
db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

old_table = 'WindSpeedPrediction'
new_table = 'IoTdata2'
default_table = old_table
id_col = 'ID'
date_col = 'date4'
voltage = 'Voltage'

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

def extract_data(cols, rows, date_col, value_names=['Current', 'WindSpeed'], other_names=[voltage]):
	line1 = rows[0]
	features = []
	for name in value_names:
		assert name in line1
		idx = line1.index(name)
		features.append([float(row[idx - 1]) for row in rows])
	other = {}
	for name in other_names:
		assert name in line1
		idx = line1.index(name)
		other[name] = np.array([float(row[idx - 1]) for row in rows]).T[::-1]
	date_idx = cols.index(date_col)
	T = [row[date_idx] for row in rows][::-1]
	XY = np.array(features).T[::-1]
	X = XY[:, :-1]
	Y = XY[:, -1]

	return X, Y, T, other

if default_table == new_table:
	select_some = select_latest_n
else:
	select_some = select_n_after_offset
cols, rows = select_some(db_connection_str, default_table, id_col, n=data_q_len)
X, Y, T, other = extract_data(cols, rows, date_col=date_col)
n_samples = len(Y)
V = other[voltage]
for i in range(n_samples):
	data_q.append((X[i:i+1, :], Y[i:i+1], T[i:i+1], V[i:i+1]))

model_fpath = 'model'
model = tf.keras.models.load_model(model_fpath)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(style={'height': '100%', 'width': '100%', 'top': 0, 'z-index': 1, 'overflow-x': 'hidden', 'padding-top': '0px', 'position': 'fixed'}, children=[

	dcc.Interval(id='interval', n_intervals=0, interval=interval * 1000),
	dbc.RadioItems(
		options=[
			{"label": 'From 11.05.2022', "value": old_table},
			{"label": 'Real time', "value": new_table},
		],
		value=default_table,
		id='data_src',
		inline=True,
		style={'padding-top': '0.5%', 'padding-bottom': '0.5%', 'padding-right': '0.5%', 'padding-left': '0.5%'}
	),
	dcc.Graph(
		id='wind_speed_plot',
		figure={},
		style={'height': '95%', 'width': '100%'}
	)

])


@callback(
	Output("wind_speed_plot", "figure"),
    Input("interval", "n_intervals"),
	Input('data_src', 'value'),
	prevent_initial_call=True
)
def update_progress(n_intervals, data_src_value):

	input_id = callback_context.triggered_id

	if input_id == 'interval':

		if n_intervals % window_step == 0:

			if data_src_value == new_table:
				cols, rows = select_latest_n(db_connection_str, data_src_value, id_col, n=window_step)
			else:
				cols, rows = select_n_after_offset(db_connection_str, data_src_value, id_col, offset=n_intervals * window_step + data_q_len, n=window_step)

			X_new, Y_new, T_new, other_new = extract_data(cols, rows, date_col=date_col)
			V_new = other_new[voltage]
			n_samples = len(Y_new)
			for i in range(n_samples):
				data_q.append((X_new[i:i + 1, :], Y_new[i:i + 1], T_new[i:i + 1], V_new[i:i + 1]))

		X = np.vstack([item[0] for item in data_q])
		Y = np.hstack([item[1] for item in data_q])
		T = np.hstack([item[2] for item in data_q])
		V = np.hstack([item[3] for item in data_q])

		idx = series_n + (n_intervals - 1) % window_step - 1
		C = X[idx, 0]
		X = X[idx - series_n + 1 : idx + 1, :].reshape(1, series_n, 1)
		Y = Y[idx]
		T = T[idx]
		V = V[idx] / 10

		P = model.predict(X, verbose=0)

		plot_q.append((Y, P, T, C, V))

	elif input_id == 'data_src':

		data_q.clear()
		plot_q.clear()

		if data_src_value == new_table:
			cols, rows = select_latest_n(db_connection_str, data_src_value, id_col, n=data_q_len)
		else:
			cols, rows = select_n_after_offset(db_connection_str, data_src_value, id_col, offset=n_intervals * interval + data_q_len, n=data_q_len)

		X, Y, T, other = extract_data(cols, rows, date_col=date_col)
		n_samples = len(Y)
		V = other[voltage]
		for i in range(n_samples):
			data_q.append((X[i:i + 1, :], Y[i:i + 1], T[i:i + 1], V[i:i + 1]))

	tick_vals, tick_texts = [], []
	if len(plot_q) > 0:
		Y = np.vstack([item[0] for item in plot_q]).reshape(-1, 1)
		P = np.hstack([item[1] for item in plot_q]).reshape(-1, 1)
		T = np.hstack([item[2] for item in plot_q]).reshape(-1, 1)
		C = np.vstack([item[3] for item in plot_q]).reshape(-1, 1)
		V = np.vstack([item[4] for item in plot_q]).reshape(-1, 1)
		I = np.arange(len(Y)).reshape(-1, 1)

		df = pd.DataFrame(np.hstack([I, Y, P, C, V]), columns=['Index', 'Real wind speed', 'Predicted wind speed', 'Current', 'Voltage (divided by 10)'])
		for i, t in enumerate(T[:, 0]):
			if t.endswith('00') and t not in tick_texts:
				tick_vals.append(i)
				tick_texts.append(t)
	else:
		df = pd.DataFrame([], columns=['Index', 'Real wind speed', 'Predicted windspeed', 'Current', 'Voltage (divided by 10)'])
	fig = px.scatter(df, x='Index', y=['Real wind speed', 'Predicted wind speed', 'Current', 'Voltage (divided by 10)'], template='plotly_white')
	fig.update_layout(xaxis={'tickmode': 'array', 'tickvals': tick_vals, 'ticktext': tick_texts})
	fig.update_layout(xaxis_range=[-1, plot_len + 1])
	fig.update_layout(xaxis_title='Date', yaxis_title='Value')
	fig.update_traces(mode='lines+markers')
	fig.update_layout(margin={'l': 20, 'b': 20, 'r': 20, 't': 20}, legend={'x': 1, 'y': 1}, legend_title_text='Metrics')

	return fig

if __name__ == '__main__':
	debug = False
	app.run_server('0.0.0.0', port=80, debug=debug, use_reloader=debug)