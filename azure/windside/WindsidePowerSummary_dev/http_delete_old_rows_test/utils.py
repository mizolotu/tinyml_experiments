import logging
import os, pyodbc

# path

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

# sql connection

server = 'tcp:jyusqlserver.database.windows.net'
database = 'IoTSQL'
driver = '{ODBC Driver 17 for SQL Server}'
username = 'jyusqlserver'
password = ''
db_connection_str = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

# database tables

src_table = 'IoTdata2'

# database columns

id_col = 'ID'

# auxiliary functions

def delete_old_values(id_first, id_last, n=6*31*24*3600):

    if id_last - id_first > n:
        n_del = id_last - id_first - n
        logging.info(n_del)
        query = f";with cte as (select top {n_del} * from {src_table} order by {id_col}) delete from cte;"
        logging.info(query)
        with pyodbc.connect(db_connection_str) as conn:
            with conn.cursor() as cursor:
               cursor.execute(query)

def select_id(which='first'):
    if which == 'first':
        query = f"select top (1) * from {src_table} order by {id_col}"
    elif which == 'last':
        query = f"select top (1) * from {src_table} order by {id_col} desc"
    else:
        raise NotImplemented        
    logging.info(query)
    rows = []
    with pyodbc.connect(db_connection_str) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            cols = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                rows.append(list(row))
    assert id_col in cols
    id_idx = cols.index(id_col)
    ids = [row.pop(id_idx) for row in rows]
    id_selected = ids[0]
    return id_selected