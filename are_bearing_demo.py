import serial, logging, cv2
import numpy as np
import pandas as pd
import os.path as osp
import dash_bootstrap_components as dbc
import plotly.express as px

from datetime import datetime
from time import time
from queue import Queue
from collections import deque
from threading import Thread
from flask import Flask, Response, send_from_directory

from dash import Dash, html, dcc, callback, Output, Input, State
from config import *

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

serial_port ='/dev/ttyACM1'
baud_rate = 115200

interval_len = 1

maxlen = 200
data_q = deque(maxlen=maxlen)

plot_len = 100
dist_q = deque(maxlen=plot_len)
loss_q = deque(maxlen=plot_len)
output_q = deque(maxlen=plot_len)

msg_len = 10
msg_q = deque(maxlen=msg_len)

def receive_vector(ser, start_marker=60, end_marker=62):

    msg = ''
    x = 'z'
    while ord(x) != start_marker:
        x = ser.read()

    while ord(x) != end_marker:
        if ord(x) != start_marker:
            msg = f'{msg}{x.decode("utf-8")}'
        x = ser.read()

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
    else:
        if 'Important: ' in msg:
            result_dict['m'] = f"{datetime.now().strftime('%d.%m.%y %H:%M:%S.%f')}: {msg.split('Important: ')[1].capitalize()}"
            print(msg)

    return result_dict, msg

def read_serial(data_q):
    ser = serial.Serial(serial_port, baud_rate)
    while(True):
        x, msg = receive_vector(ser)
        if x is not None and len(x) > 0:
            #print(x)
            data_q.append(x)
            #data_q.put(x)

read_serial_thr = Thread(target=read_serial, args=(data_q,), daemon=False)
read_serial_thr.start()

server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # 800 x 1280
        if image is not None:
            image = image[10:720, 90:1170]
            ret, jpeg = cv2.imencode('.jpg', image)
            result = jpeg.tobytes()
        else:
            result = None
        return result

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video0')
def video0():
    print('here')
    return send_from_directory('/dev', 'video0')

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div(className='full', children=[

    dcc.Store(id='b', data=None),
    dcc.Store(id='c', data=None),
    dcc.Store(id='thr', data=None),

    dcc.Interval(id='interval', n_intervals=0, interval=interval_len * 1000),

    dbc.Row(justify='start', className='padded', style={'height': '100.0%'}, children=[

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 8}, children=[

            dbc.Row(justify='start', className='padded', style={'height': '38.4%'}, children=[

                html.H4('Logs:', style={'width': '100%', 'height': '10%'}),

                dbc.Textarea(
                    id='log_area',
                    value='',
                    style={'color': 'black', 'height': '90%', 'width': '100%'},
                    rows=msg_len,
                )

            ]),

            dbc.Row(justify='start', className='padded', style={'height': '28.4%'}, children=[

                html.H4('Anomaly detection:', style={'height': '10%'}),

                dcc.Graph(
                    id='loss_plot',
                    figure={},
                    style={'height': '90%', 'width': '100%'}
                )

            ]),

            dbc.Row(justify='start', className='padded', style={'height': '28.4%'}, children=[

                html.H4('Vibration detection:', style={'height': '10%'}),

                dcc.Graph(
                    id='dist_plot',
                    figure={},
                    style={'height': '90%', 'width': '100%'}
                )

            ])

        ]),

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 4}, children=[

            #html.Video(
            #    src='/video_feed',
            #    controls=True,
            #    #autoPlay=True,
            #    style={'height': '100%', 'width': '100%'}
            #)

            dbc.Row(justify='start', className='padded', style={'height': '38.4%'}, children=[

                html.H4('Live image:', style={'height': '10%', 'width': '100%'}),

                html.Img(
                    src="/video_feed",
                    style={'height': '90%', 'width': '100%'}
                )

            ]),

            dbc.Row(justify='start', className='padded', style={'height': '58.4%'}, children=[

                html.H4('Output space of the model:', style={'height': '10%', 'width': '100%'}),

                dcc.Graph(
		            id='output_plot',
		            figure={},
		            style={'height': '90%', 'width': '100%'}
	            )

            ])

        ])

    ])
])

@callback(
    Output("log_area", "value"),
    Output('b', 'data'),
    Output('c', 'data'),
    Output('thr', 'data'),
    Output("dist_plot", "figure"),
    Output("loss_plot", "figure"),
    Output("output_plot", "figure"),

    Input("interval", "n_intervals"),

    State('b', 'data'),
    State('c', 'data'),
    State('thr', 'data'),
    State("dist_plot", "figure"),
    State("loss_plot", "figure"),
    State("output_plot", "figure"),
    prevent_initial_call=True
)
def update_progress(n_intervals, b_data_state, c_data_state, thr_data_state, dist_fig_state, loss_fig_state, output_fig_state):

    items = [data_q.popleft() for _ in range(len(data_q))]

    # logs

    messages = [item['m'] for item in items if 'm' in item.keys()]
    for msg in messages:
        if msg not in list(msg_q):
            msg_q.append(msg)

    log_area_value = '\n'.join([item for item in msg_q])

    # b

    b_data = b_data_state
    if b_data_state is None:
        bs = [item['b'] for item in items if 'b' in item.keys()]
        if len(bs) > 0:
            b_data = bs[0]
            print(f'Baseline threshold = {b_data}')

    # c

    c_data = c_data_state
    if c_data_state is None:
        cs = [item['c'] for item in items if 'c' in item.keys()]
        if len(cs) > 0:
            c_data = cs[0]
            print(f'Model centre = {c_data}')

    # thr

    thr_data = thr_data_state
    if thr_data_state is None:
        thrs = [item['t'] for item in items if 't' in item.keys()]
        if len(thrs) > 0:
            thr_data = thrs[0]
            print(f'Score threshold = {thr_data}')

    # xyz dists

    dists = [item['d'] for item in items if 'd' in item.keys()]
    if len(dists) > 0:
        # loss_q.append([np.mean(losses), np.max(losses)])
        dist_q.append(np.max(dists))
        # loss_q.append(np.mean(losses))
        dists = [item for item in dist_q]
    else:
        dists = []

    if len(dists) > 0:
        idx = np.arange(np.maximum(0, n_intervals - len(dists)), n_intervals).reshape(-1, 1)
        if b_data is not None:
            losses = np.hstack([idx, np.vstack(dists), np.ones(len(dists)).reshape(-1, 1) * b_data[0]])
            y = ['Maximal', 'Threshold']
        else:
            losses = np.hstack([idx, np.vstack(dists)])
            y = ['Maximal']
        df = pd.DataFrame(losses, columns=['Time', *y])
        dist_fig = px.scatter(df, x='Time', y=y, template='plotly_white', color_discrete_sequence=['#1616A7', '#FB0D0D'])
        dist_fig.update_layout(xaxis_title='Time since the start, seconds', yaxis_title='Value')
        dist_fig.update_xaxes(range=[np.min(idx), np.min(idx) + plot_len])
        #loss_fig.update_yaxes(range=[0 - 0.01, 0.25 + 0.01])
        #loss_fig.update_yaxes(range=[0 - 0.01, 10.0 + 0.01])
        dist_fig.update_traces(mode='lines+markers')
        dist_fig.update_layout(margin={'l': 20, 'b': 20, 'r': 20, 't': 20}, legend={'x': 1, 'y': 1}, legend_title_text='Distance:')
    else:
        dist_fig = dist_fig_state


    # losses

    losses = [item['l'] for item in items if 'l' in item.keys()]
    if c_data is not None and len(losses) > 0:
        #loss_q.append([np.mean(losses), np.max(losses)])
        loss_q.append(np.max(losses))
        # loss_q.append(np.mean(losses))
        losses = [item for item in loss_q]
    else:
        losses = []

    labels = np.array([1 if thr_data is not None and l > thr_data[0] else 0 for l in losses])

    if len(losses) > 0:
        idx = np.arange(np.maximum(0, n_intervals - len(losses)), n_intervals).reshape(-1, 1)
        if thr_data is not None:
            losses = np.hstack([idx, np.vstack(losses), np.ones(len(losses)).reshape(-1, 1) * thr_data[0]])
            y = ['Maximal', 'Threshold']
        else:
            losses = np.hstack([idx, np.vstack(losses)])
            y = ['Maximal']
        df = pd.DataFrame(losses, columns=['Time', *y])
        loss_fig = px.scatter(df, x='Time', y=y, template='plotly_white', color_discrete_sequence=['#1616A7', '#FB0D0D'])
        loss_fig.update_layout(xaxis_title='Time since the start, seconds', yaxis_title='Value')
        loss_fig.update_xaxes(range=[np.min(idx), np.min(idx) + plot_len])
        #loss_fig.update_yaxes(range=[0 - 0.01, 0.25 + 0.01])
        #loss_fig.update_yaxes(range=[0 - 0.01, 10.0 + 0.01])
        loss_fig.update_traces(mode='lines+markers')
        loss_fig.update_layout(margin={'l': 20, 'b': 20, 'r': 20, 't': 20}, legend={'x': 1, 'y': 1}, legend_title_text='Loss:')
    else:
        loss_fig = loss_fig_state

    # outputs

    outputs = [item['o'] for item in items if 'o' in item.keys()]

    #print(len(labels), len(outputs))

    for i, o in enumerate(outputs):
        if thr_data is not None and c_data is not None and np.mean((np.array(c_data) - np.array(o)) ** 2) > 2 * thr_data[0] and labels[-1] == 1:
            label = 1
            size = 7.5
        else:
            label = 0
            size = 5.0
        output_q.append([*o, label, size])

    outputs = [output_q.popleft() for _ in range(len(output_q))]

    if c_data is not None:
        outputs.append([*c_data, -1, 10.0])
    elif len(outputs) > 0:
        #print(np.mean([item[:2] for item in outputs], 0))
        outputs.append([*np.mean([item[:2] for item in outputs], 0), -1, 10.0])

    if len(outputs) > 0:
        df = pd.DataFrame(outputs, columns=['x', 'y', 'label', 'size']).sort_values(by=['label'])
        df["label"] = df["label"].astype(str)
        output_fig = px.scatter(df, x='x', y='y', color='label', size='size', template='plotly_white', color_discrete_sequence=['#222A2A', '#1616A7', '#FB0D0D'])
        #output_fig.update_xaxes(range=[-0.21, 1.21])
        #output_fig.update_yaxes(range=[-0.21, 1.21])
        if c_data is not None:
            output_fig.update_xaxes(range=[c_data[0] - 15, c_data[0] + 15])
            output_fig.update_yaxes(range=[c_data[1] - 15, c_data[1] + 15])
            if thr_data is not None:
                output_fig.update_xaxes(range=[c_data[0] - 3 * np.sqrt(2 * thr_data[0]), c_data[0] + 3 * np.sqrt(2 * thr_data[0])])
                output_fig.update_yaxes(range=[c_data[1] - 3 * np.sqrt(2 * thr_data[0]), c_data[1] + 3 * np.sqrt(2 * thr_data[0])])
                output_fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=c_data[0] - np.sqrt(2 * thr_data[0]),
                    y0=c_data[1] - np.sqrt(2 * thr_data[0]),
                    x1=c_data[0] + np.sqrt(2 * thr_data[0]),
                    y1=c_data[1] + np.sqrt(2 * thr_data[0]),
                    line_color='#1616A7',
                    fillcolor='#1616A7',
                    opacity = 0.2
                )
        output_fig.update_layout(xaxis_title='Feature 1', yaxis_title='Feature 2')
        output_fig.update_traces(mode='markers')
        output_fig.update_layout(margin={'l': 20, 'b': 20, 'r': 20, 't': 20}, legend={'x': 1, 'y': 1}, legend_title_text='Samples:')
        #output_fig.update_layout(margin={'l': 20, 'b': 20, 'r': 20, 't': 20}, legend_title_text='Samples:')
        names = {'0': 'Normal', '-1': 'Center', '1': 'Anomaly'}
        for i in range(len(output_fig.data)):
            output_fig.data[i].name = names[output_fig.data[i].name]
    else:
        output_fig = output_fig_state

    return log_area_value, b_data, c_data, thr_data, dist_fig, loss_fig, output_fig

if __name__ == '__main__':
    debug = False
    app.run_server(debug=debug, use_reloader=debug)