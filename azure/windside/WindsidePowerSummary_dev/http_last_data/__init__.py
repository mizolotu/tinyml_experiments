import logging, sys, os
import azure.functions as func

from .utils import prepare_summary, generate_html

def main(req: func.HttpRequest) -> func.HttpResponse:

    lookback = None
    params = req.params    
    for key, value in params.items():
        if key == 'hours':
            try:
                lookback = float(value) * 3600
            except:
                pass

    try:
        ts_first, ts_last, summary_lines, bars = prepare_summary(lookback=lookback)        
        html = generate_html(ts_first, ts_last, summary_lines, bars)

        return func.HttpResponse(html, mimetype='text/html')

    except Exception as e:
        logging.info(e)
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return func.HttpResponse(f'Exception {e} in file {fname} at line {exc_tb.tb_lineno}', status_code=500)   