import logging, sys, os
import azure.functions as func

from .utils import select_last_id, select_latest_n_samples, insert_new_values

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        id_last = select_last_id()        
        #logging.info(id_last)
        cols_new, rows_new = select_latest_n_samples(n=3600, id=id_last)
        if len(rows_new) > 0:
            insert_new_values(cols_new, rows_new)
        return func.HttpResponse('ok')
    except Exception as e:
        logging.info(e)
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return func.HttpResponse(f'Exception {e} in file {fname} at line {exc_tb.tb_lineno}', status_code=500)



    
