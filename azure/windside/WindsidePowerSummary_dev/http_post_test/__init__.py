import logging, sys, os
import azure.functions as func

from .utils import generate_html

def main(req: func.HttpRequest) -> func.HttpResponse:

    try:
        html = generate_html(req.url, req.params)
        return func.HttpResponse(html, mimetype='text/html')

    except Exception as e:
        logging.info(e)
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return func.HttpResponse(f'Exception {e} in file {fname} at line {exc_tb.tb_lineno}', status_code=500)   