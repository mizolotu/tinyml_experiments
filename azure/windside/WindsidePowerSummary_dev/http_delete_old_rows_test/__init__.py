import logging, sys, os
import azure.functions as func

from .utils import select_id, delete_old_values

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        id_first = select_id('first')
        logging.info(id_first)
        id_last = select_id('last')
        logging.info(id_last)
        delete_old_values(id_first, id_last, n=16070400)
        return func.HttpResponse('ok')
    except Exception as e:
        logging.info(e)
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return func.HttpResponse(f'Exception {e} in file {fname} at line {exc_tb.tb_lineno}', status_code=500)



    
