import os.path as osp

import numpy as np
import pandas as pd

from datetime import datetime

from config import WINDSIDE_DATA_DIR
from windside_download_data_from_azure import db_connection_str, download_in_chunks
from sklearn.linear_model import LinearRegression as LR

if __name__ == '__main__':

    lora_table = 'WindsideIoT'
    id_col = 'ID'
    lte_timestamp_col = 'Timestamp'
    lora_date_col = 'date'
    voltage_col = 'Voltage'

    lte_fpath = osp.join(WINDSIDE_DATA_DIR, 'features.csv')
    lora_fpath = osp.join(WINDSIDE_DATA_DIR, 'lora_data.csv')

    if not osp.isfile(lora_fpath):
        cols, rows = download_in_chunks(lora_table, id_col, lora_date_col, chunk_size=200000)
        pd.DataFrame(rows, columns=cols).to_csv(lora_fpath, index=False)

    lora_df = pd.read_csv(lora_fpath)
    lora_dates = lora_df[lora_date_col].values
    lora_voltages = lora_df[voltage_col].values
    lora_ts = np.array([datetime.strptime(item.split('.')[0], '%Y-%m-%d %H:%M:%S').timestamp() for item in lora_dates])
    lora_idx = np.argsort(lora_ts)
    lora_ts_sorted = lora_ts[lora_idx]
    lora_voltages_sorted = lora_voltages[lora_idx]

    lora_interval = [
        datetime.strptime('2023-05-01 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp(),
        datetime.strptime('2023-08-16 00:00:00', '%Y-%m-%d %H:%M:%S').timestamp(),
    ]

    lora_idx = np.where((lora_ts_sorted >= lora_interval[0]) & (lora_ts_sorted <= lora_interval[1]))
    lora_ts = lora_ts_sorted[lora_idx]
    lora_voltages = lora_voltages_sorted[lora_idx]

    lte_df = pd.read_csv(lte_fpath)
    lte_ts = lte_df[lte_timestamp_col].values
    lte_voltages = lte_df[voltage_col].values

    lte_idx = np.where((lte_ts >= lora_ts[0]) & (lte_ts <= lora_ts[-1]))[0]
    lte_ts = lte_ts[lte_idx]
    lte_voltages = lte_voltages[lte_idx]

    lora_voltages_, lte_voltages_ = [], []
    lora_voltages_sum, lte_voltages_sum = 0, 0
    n_voltages = 0
    count = 0

    for i in range(len(lora_ts) - 1):
        if (lora_ts[i+1] - lora_ts[i]) >= 57  and (lora_ts[i+1] - lora_ts[i]) <= 63:
            idx = np.where((lte_ts >= lora_ts[i]) & (lte_ts <= lora_ts[i+1]))[0]
            if len(idx) >= 57 and len(idx) <= 63:
                lora_voltages_sum += lora_voltages[i + 1]
                lte_voltages_sum += np.mean(lte_voltages[idx])
                n_voltages += 1
                lora_voltages_.append(lora_voltages[i+1])
                lte_voltages_.append(np.mean(lte_voltages[idx]))
                count += 1
                if i % 1000 == 0:
                    X = np.array(lora_voltages_).reshape(-1, 1)
                    y = np.array(lte_voltages_)
                    lr = LR()
                    lr.fit(X, y)
                    print(i, count, lr.coef_, lr.predict([[0]]), (lte_voltages_sum - lora_voltages_sum) / n_voltages)

