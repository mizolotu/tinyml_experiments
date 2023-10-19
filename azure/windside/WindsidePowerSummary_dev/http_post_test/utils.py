import logging
import os, pyodbc, base64, io

from matplotlib import pyplot as pp
from datetime import datetime

# path

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

# auxiliary functions

def generate_html(url, params):

    logging.info(url)
    logging.info(params)

    if 'code' in params.keys():
        code = params["code"]
    else:
        code = None

    logging.info(code)

    sdate = '11/05/2022 00:00:00'
    if 'sdate' in params.keys():
        try:
            sdate = params['sdate']            
        except Exception as e:
            logging.info(e)
            
    edate = '16/05/2023 23:59:59'
    if 'edate' in params.keys():
        try:
            edate = params['edate']            
        except Exception as e:
            logging.info(e)

    sdate_ut = datetime.strptime(sdate, '%d/%m/%Y %H:%M:%S').timestamp()
    edate_ut = datetime.strptime(edate, '%d/%m/%Y %H:%M:%S').timestamp()

    logging.info(sdate_ut)
    logging.info(edate_ut)

    form_tag = [
        f'<form action="{url}">',
        f'<label for="sdate">Start date:</label><br><input type="text" id="sdate" name="sdate" value="{sdate}"><br>',
        f'<label for="edate">End date:</label><br><input type="text" id="edate" name="edate" value="{edate}"><br>',
        f'<input type="hidden" id="code" name="code" value="{code}" />',
        '<br><input type="submit" value="Submit"></form>'
    ]
    form_tag = ''.join(form_tag)    

    # interval
            
    interval_line = f'Interval: {sdate} - {edate}'

    html = f'<html><head></head><body>{form_tag}<p>{interval_line}</p></body></html>'

    return html