import logging, sys, os
import azure.functions as func

from .utils import select_current_summary, select_latest_n_samples, update_summary

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:

        id_last, cols_summary, rows_summary = select_current_summary()        
        logging.info(id_last)
        logging.info(cols_summary)
        logging.info(rows_summary)
        cols_new, rows_new = select_latest_n_samples(n=100, id=id_last)
        if len(rows_new) > 0:
            update_summary(cols_summary, rows_summary, cols_new, rows_new)

        return func.HttpResponse('ok')

    except Exception as e:
        logging.info(e)
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return func.HttpResponse(f'Exception {e} in file {fname} at line {exc_tb.tb_lineno}', status_code=500)



    
