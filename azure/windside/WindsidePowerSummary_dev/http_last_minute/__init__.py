import logging, sys, os
import azure.functions as func

from .utils import select_latest_n_samples, generate_summary

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        cols_new, rows_new = select_latest_n_samples(n=60)
        if len(rows_new) > 0:
            lines = generate_summary(cols_new, rows_new, lookback=60)

        return func.HttpResponse(lines)

    except Exception as e:
        logging.info(e)
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return func.HttpResponse(f'Exception {e} in file {fname} at line {exc_tb.tb_lineno}', status_code=500)



    
