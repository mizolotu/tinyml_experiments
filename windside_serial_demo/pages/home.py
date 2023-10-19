import os, json
import os.path as osp
import numpy as np

import dash_bootstrap_components as dbc
import plotly.express as px

from dash import html, Input, State, Output, dcc, register_page, callback, ALL, callback_context

register_page(__name__, path='/')

layout = html.Div(className='full', children=[

    dbc.Row(className='padded', justify='start', align="center", style={'height': '20%'}, children=[

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 10}, children=[

            html.H3('Introduction'),

            dcc.Markdown(
                '''
                Finland-based Oy Windside Production Ltd. (Windside) is a specialized manufacturer of robust, rugged, and reliable vertical axis wind turbine (VAWT) systems.
                The unique "helix" shape was studied & refined for use as a wind turbine by Risto Joutsiniemi in 1978. It incorporates many unique & patented elements which 
                result in an evolutionary leap from the classic Savonious turbine design.
                
                Windside wind turbine is made for professional use but found also by artist, architects and designers all over the world. You can find Windside turbines in 
                mountains thousands of meters above sea level; on glaciers; on coastlines; on deep sea and marine navigation systems; in deserts, in remote rural areas. But 
                also in densely populated urban areas! You will find tens of examples described on [the company web page](https://windside.com). Take your time and enjoy!
                '''
            )

        ]),

        dbc.Col(align='center', children=[

            dbc.Button(
                'Go to demo',
                href='demo',
                style={'width': '100%'}
            )

        ])

    ]),

    dbc.Row(className='padded', justify='start', align='center', style={'height': '40%'}, children=[

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 4}, children=[

            html.Img(
                src='assets/wind_tourbine.jpg', style={'height': '100%'}
            )

        ]),

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 4}, children=[

            html.Img(
                src='assets/lora_signal.png', style={'height': '100%'}
            )

        ]),

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 4}, children=[

            html.Img(
                src='assets/engineers.jpg', style={'height': '100%'}
            )

        ]),

    ]),

    dbc.Row(className='padded', justify='start', style={'height': '40%'}, children=[

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 12}, children=[

            html.Img(src='assets/demo.png', style={'width': '100%'}),

        ]),

    ]),

])