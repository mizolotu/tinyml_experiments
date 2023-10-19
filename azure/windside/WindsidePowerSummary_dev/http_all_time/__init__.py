import logging, sys, os
import azure.functions as func

from .utils import select_current_summary, prepare_summary

def main(req: func.HttpRequest) -> func.HttpResponse:    
    try:
        _, cols, rows = select_current_summary()
        summary = prepare_summary(cols, rows)            
        return func.HttpResponse(summary)
    except Exception as e:
        logging.info(e)
        _, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.info(f"Exception {e} in file {fname} at line {exc_tb.tb_lineno}")
    