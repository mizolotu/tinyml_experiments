import dash_bootstrap_components as dbc

from dash import Dash, html, page_container

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
	page_container
])

if __name__ == '__main__':
	app.run_server('0.0.0.0', debug=False, use_reloader=False)