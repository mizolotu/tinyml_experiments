import json
import pandas as pd
import numpy as np

from config import *
from datetime import datetime
from time import time
from scipy import stats

def get_last_line(fpath):
    with open(fpath, "rb") as file:
        try:
            file.seek(-2, os.SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, os.SEEK_CUR)
        except OSError:
            file.seek(0)
        last_line = file.readline().decode().strip()
    return last_line

def extract_data(input_dir, output_fpath, chunk_size=None, start_date='10/05/2022 23:59:59',
                 value_names=['Current', 'Voltage', 'WindSpeed'], id_col='ID', date_col='date4'):

    data = []

    try:

        last_line = get_last_line(output_fpath)
        start_date_dt = datetime.strptime(last_line.split(',')[1], '%d/%m/%Y %H:%M:%S')
        append = True

    except Exception as e:
        print(e)
        start_date_dt = datetime.strptime(start_date, '%d/%m/%Y %H:%M:%S')
        append = False

    start_date_timestamp = start_date_dt.timestamp()

    print('Last timestamp in the output:', start_date, start_date_dt, start_date_timestamp)

    #date_parser = lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

    input_fnames = sorted(os.listdir(input_dir))

    input_fnames_to_extract = []
    for input_fname in input_fnames:
        input_fpath = osp.join(input_dir, input_fname)
        df = pd.read_csv(input_fpath, nrows=1)
        dates = df[date_col].values
        timestamp0 = datetime.strptime(dates[0], '%d/%m/%Y %H:%M:%S').timestamp()
        cols = [key for key in df.keys()]
        date_col_idx = cols.index(date_col)
        last_line_spl = get_last_line(input_fpath).split(',')
        timestamp1 = datetime.strptime(last_line_spl[date_col_idx], '%d/%m/%Y %H:%M:%S').timestamp()

        #print(input_fpath, dates[0], timestamp0, last_line_spl[date_col_idx], timestamp1)

        if timestamp0 > start_date_timestamp or timestamp1 > start_date_timestamp:
            input_fnames_to_extract.append(input_fname)

    print('Extracting data from:', input_fnames_to_extract)

    header = None

    timestamp_col = 'Timestamp'

    for input_fname in input_fnames_to_extract:

        #if not input_fname.startswith('11'):
        #    continue

        print('Extracting data from:', input_fname)

        input_fpath = osp.join(input_dir, input_fname)

        if chunk_size is not None:
            df = pd.read_csv(input_fpath, chunksize=chunk_size)
            df = next(df)
        else:
            df = pd.read_csv(input_fpath)

        df = df.sort_values(by=[id_col])

        dates = df[date_col].values
        #seconds = (dates.astype(datetime) * 1e-9).astype(float)
        t_start = time()
        seconds = np.array([datetime.strptime(d, '%d/%m/%Y %H:%M:%S').timestamp() for d in dates])
        print('Converting date to timestamp took', time() - t_start, 'seconds')

        #start_date = pd.to_datetime(start_date, format='%Y-%d-%m %H:%M:%S')

        #idx = [i for i, date1 in enumerate(dates) if date1 == start_date]
        #idx = [i for i, (date1, date2) in enumerate(zip(dates[:-1], dates[1:])) if pd.to_datetime(date1).year == 2022 and pd.to_datetime(date2).year == 2023]
        #print(idx)
        #print(df.values[idx, :])

        row_idx = np.where(seconds > start_date_timestamp)[0]
        #row_idx = np.where(seconds >= start_date_dt.timestamp())[0]
        print('Row index range:', row_idx[0], '-', row_idx[-1])
        print('Date range:', dates[row_idx[0]], '-', dates[row_idx[-1]])

        d_delta = np.array((seconds[row_idx][1:] - seconds[row_idx][:-1]), dtype=float)
        print('Max time delta =', np.max(d_delta), 'mean time delta =', np.mean(d_delta))

        start_date_dt = dates[-1]
        start_date_timestamp = seconds[-1]

        print('New last date:', start_date_dt, start_date_timestamp)

        if header is None:
            header = [key for key in df.keys()]
        else:
            assert header == [key for key in df.keys()]

        line1 = df.values[0, :]
        feature_cols = []
        for name in value_names:
            assert name in line1
            idx = np.where(line1 == name)[0][0]
            feature_cols.append(header[idx - 1])

        selected_cols = [id_col] + [date_col] + feature_cols

        data.append(np.hstack([df[selected_cols].values[row_idx, :], seconds[row_idx].reshape(-1, 1)]))

    data = np.vstack(data)
    print('Data shape:', data.shape)

    header = [id_col] + ['Date'] + value_names + [timestamp_col]

    df_new = pd.DataFrame(data)

    if append:
        df_new.to_csv(output_fpath, index=False, header=False, mode='a')
    else:
        df_new.to_csv(output_fpath, index=False, header=header)

def create_dataset(input_fpath, output_fpath_pattern=None, feature_cols=['Current'], label_col='WindSpeed',
                   timestamp_col='Timestamp', remove_negative_voltage=True, series_step=1, window_size=60,
                   window_step=1, chunk_size=None, label_mode='avg+max', d_delta_thr=3, voltage_col='Voltage',
                   date_col='Date', feature_extractor='raw', weights='discrete', gaussian_kde_n_samples=1000):

    dataset_exists = False

    file_stats = os.stat(input_fpath)
    input_fsize = file_stats.st_size

    if output_fpath_pattern is not None:
        try:
            with open(output_fpath_pattern.format('metainfo', 'json'), 'r') as f:
                metainfo = json.load(f)

            file_stats = os.stat(output_fpath_pattern.format('dataset_xyw', 'csv'))
            output_fsize = file_stats.st_size

            if 'input_fsize' in metainfo.keys() and metainfo['input_fsize'] == input_fsize:
                if 'output_fsize' in metainfo.keys() and metainfo['output_fsize'] == output_fsize:
                    if 'chunk_size' in metainfo.keys() and metainfo['chunk_size'] == chunk_size:
                        if 'feature_extractor' in metainfo.keys() and metainfo['feature_extractor'] == feature_extractor:
                            if 'series_step' in metainfo.keys() and metainfo['series_step'] == series_step:
                                dataset_exists = True
        except Exception as e:
            print(e)

    print('Dataset exists:', dataset_exists)

    if dataset_exists:

        df_xyw = pd.read_csv(output_fpath_pattern.format('dataset_xyw', 'csv'), header=None)
        df_td = pd.read_csv(output_fpath_pattern.format('dataset_td', 'csv'), header=None)

        vals_xyw = df_xyw.values
        vals_td = df_td.values

        input_dim = metainfo['input_dim']
        output_dim = metainfo['output_dim']
        n_series = vals_xyw.shape[0]

        X = (vals_xyw[:, : np.prod(input_dim)].reshape(n_series, *input_dim)).astype(np.float32)
        Y_ = (vals_xyw[:, np.prod(input_dim) : np.prod(input_dim) + np.prod(output_dim)]).astype(np.float32)
        W_ = (vals_xyw[:, np.prod(input_dim) + np.prod(output_dim) :]).astype(np.float32)

        T = (vals_td[:, :4]).astype(np.float32)
        D = (vals_td[:, 4:].flatten()).astype('object')

        n_samples = X.shape[0]

        if weights == 'discrete':
            W = W_[:, 0]
        elif weights == 'continuous':
            W = W_[:, 1]
        elif weights is None:
            W = np.ones(n_samples)
        else:
            raise NotImplemented

        if label_mode == 'avg':
            Y = Y_[:, [0]]
        elif label_mode == 'max':
            Y = Y_[:, [1]]
        elif label_mode == 'avg+max':
            Y = Y_
        else:
            raise NotImplemented

        y0_mean = np.array(metainfo['y0_mean'])
        y0_error = np.array(metainfo['y0_error'])

    else:

        if chunk_size is not None:
            df = pd.read_csv(input_fpath, chunksize=chunk_size)
            df = next(df)
        else:
            df = pd.read_csv(input_fpath)

        print('Min:', np.min(df.values[np.where(df.values[:, 2] > 0)[0], :], 0))
        print('Max:', np.max(df.values[np.where(df.values[:, 2] > 0)[0], :], 0))

        if remove_negative_voltage:
            idx = np.where(df[voltage_col] >= 0)[0]
        else:
            idx = np.arange(df.shape[0])

        series_dim = len(feature_cols)
        input_dim = [window_size // series_step, series_dim]
        output_dim = [2]

        day = 24 * 60 * 60
        year = (365.2425) * day

        dates = df[date_col].values[idx]
        seconds = df[timestamp_col].values[idx]

        features = (df[feature_cols].values[idx]).astype(np.float32)
        labels = (df[label_col].values[idx]).astype(np.float32)

        d_delta = np.array(seconds[1:] - seconds[:-1], dtype=float)

        idx_big_delta = np.where(d_delta > d_delta_thr)[0]

        X, Y, T, D = [], [], [], []
        Y0 = []

        offset = 0

        while len(idx_big_delta) > 0:

            print(f'Progress: {offset}/{len(d_delta)}')

            upper = offset + idx_big_delta[0]

            n = upper - offset

            n_series = int((n - window_size - series_step + 1) / window_step)

            if n_series > 0:

                if feature_extractor == 'raw':
                    X_ = np.zeros((n_series, *input_dim))
                elif feature_extractor == 'pam':
                    X_ = np.zeros((n_series, 4, series_dim))
                else:
                    raise NotImplemented

                T_ = np.zeros((n_series, 4), dtype=np.float32)
                Y_ = np.zeros((n_series, 2), dtype=np.float32)
                D_ = np.zeros(n_series, dtype=object)

                for i in range(0, n_series):
                    x_idx = offset + i * window_step
                    T_[i, 0] = np.sin(seconds[x_idx + window_size] * (2 * np.pi / day))
                    T_[i, 1] = np.cos(seconds[x_idx + window_size] * (2 * np.pi / day))
                    T_[i, 2] = np.sin(seconds[x_idx + window_size] * (2 * np.pi / year))
                    T_[i, 3] = np.cos(seconds[x_idx + window_size] * (2 * np.pi / year))

                for i in range(0, n_series):
                    x_idx = offset + i * window_step
                    x_idx_range = np.arange(x_idx + series_step, x_idx + window_size + series_step, series_step)

                    if feature_extractor == 'raw':
                        X_[i, :, :] = features[x_idx_range, :]
                    elif feature_extractor == 'pam':
                        X_[i, 0, :] = np.mean(features[x_idx_range, :])
                        X_[i, 1, :] = np.std(features[x_idx_range, :])
                        X_[i, 2, :] = np.min(features[x_idx_range, :])
                        X_[i, 3, :] = np.max(features[x_idx_range, :])
                    else:
                        raise NotImplemented

                    Y_[i, 0] = np.mean(labels[x_idx_range])
                    Y_[i, 1] = np.max(labels[x_idx_range])

                    D_[i] = dates[x_idx + window_size]

                X_sum = np.array([np.sum(x) for x in X_])
                idx0 = np.where(X_sum == 0)[0]
                idx1 = np.where(X_sum != 0)[0]

                Y0.append(Y_[idx0, :])

                X.append(X_[idx1, :])
                Y.append(Y_[idx1])
                T.append(T_[idx1, :])
                D.append(D_[idx1])

            idx_small_delta = np.where(d_delta[upper:] <= d_delta_thr)[0]
            if len(idx_small_delta) > 0:
                offset = upper + idx_small_delta[0]
                idx_big_delta = np.where(d_delta[offset:] > d_delta_thr)[0]
            else:
                break

        del df, features, labels, dates, seconds

        Y0 = np.vstack(Y0)
        y0_n = Y0.shape[0]
        y0_mean = np.mean(Y0, axis=0)
        y0_error = np.mean(np.abs(Y0 - y0_mean), axis=0)

        X = np.vstack(X).astype(np.float32)
        T = np.vstack(T).astype(np.float32)
        Y = np.vstack(Y).astype(np.float32)
        D = np.hstack(D).astype('object')

        if feature_extractor == 'raw':
            X_ = X[:, :, 0]  # assuming the current is always first
            C_mean = np.mean(X_, 1)
        elif feature_extractor == 'pam':
            C_mean = X[:, 0, 0]  # assuming the current is always first

        C_mean_levels = C_mean // 0.1 * 0.1

        C_mean_levels_u = np.unique(C_mean_levels)
        C_mean_levels_n = [len(np.where(C_mean_levels == level)[0]) for level in C_mean_levels_u]
        n_samples = len(C_mean)

        print('Calculating weights...')

        n_points_to_sample = np.minimum(len(C_mean), gaussian_kde_n_samples)
        C_mean_samples = np.random.choice(C_mean, n_points_to_sample)
        density = stats.gaussian_kde(C_mean_samples)
        dens = density.evaluate(C_mean)

        #print('KS-test:', stats.kstest(C_mean, density))

        W_continuous = 1 / ((np.max(C_mean) - np.min(C_mean)) ** 2) / np.maximum(dens ** 2, 1 / n_samples)
        W_continuous = W_continuous.reshape((n_samples, 1))

        W_discrete = np.zeros((n_samples, 1))
        for i in range(n_samples):
            j = np.where(C_mean_levels_u==C_mean_levels[i])[0][0]
            W_discrete[i] = n_samples / len(C_mean_levels_u) / C_mean_levels_n[j]

        #pp.plot(C_mean, W_discrete, 'bo')
        #pp.plot(C_mean, W_continuous, 'rs')
        #pp.savefig('weights.png')
        #pp.close()

        W = np.hstack([W_discrete, W_continuous]).astype(np.float32)

        if output_fpath_pattern is not None:

            n_series = X.shape[0]
            print('Number of series in the dataset:', n_series)

            df_td = pd.DataFrame(np.hstack([
                T.reshape(n_series, 4),
                D.reshape(n_series, -1)
            ]))

            print(f'(T,D)-frame shape: {df_td.shape}')

            output_fpath_td = output_fpath_pattern.format('dataset_td', 'csv')
            df_td.to_csv(output_fpath_td, header=None, index=None)

            df_xyw = pd.DataFrame(np.hstack([
                X.reshape(n_series, np.prod(input_dim)),
                Y.reshape(n_series, np.prod(output_dim)),
                W.reshape(n_series, -1)
            ]).astype(np.float32))

            print(f'(X,Y,W)-frame shape: {df_xyw.shape}')

            output_fpath_xyw = output_fpath_pattern.format('dataset_xyw', 'csv')
            df_xyw.to_csv(output_fpath_xyw, header=None, index=None)

            file_stats = os.stat(output_fpath_pattern.format('dataset_xyw', 'csv'))
            output_fsize = file_stats.st_size

            metainfo = {
                'input_fsize': input_fsize,
                'output_fsize': output_fsize,
                'chunk_size': chunk_size,
                'label_mode': label_mode,
                'feature_extractor': feature_extractor,
                'n_samples': n_series,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'series_step': series_step,
                'y0_n': float(y0_n),
                'y0_mean': y0_mean.tolist(),
                'y0_error': y0_error.tolist(),
            }
            with open(output_fpath_pattern.format('metainfo', 'json'), 'w') as f:
                json.dump(metainfo, f)

        X = X.reshape(n_series, *input_dim)

        print(f'Shapes: X = {X.shape}, Y = {Y.shape}, W = {W.shape}, T = {T.shape}, D = {D.shape}')

        if weights == 'discrete':
            W = W[:, 0]
        elif weights == 'continuous':
            W = W[:, 1]
        elif weights is None:
            W = np.ones(n_samples)
        else:
            raise NotImplemented

    return X, T, Y, D, W, y0_mean, y0_error

def split_dataset(data=None, input_fpath_pattern=None, chunk_size=None, val_split=0.2, inf_split=0.1, order='random', shuffle_train=False):

    if data is not None and len(data) == 5:
        X = data[0]
        T = data[1]
        Y = data[2]
        D = data[3]
        W = data[4]

    elif data is None and input_fpath_pattern is not None:

        raise NotImplemented

        if chunk_size is not None:
            df = pd.read_csv(input_fpath_pattern.format('dataset'), header=None, chunksize=chunk_size)
            df = next(df)
        else:
            df = pd.read_csv(input_fpath_pattern.format('dataset'), header=None)

        with open(input_fpath_pattern.format('metainfo'), 'r') as f:
            meta = json.load(f)

        input_dim = meta['input_dim']
        output_dim = meta['output_dim']

        values = df.values
        X = values[:, :-1].reshape([-1, *input_dim])
        Y = values[:, -1].reshape([-1, *output_dim])

    else:
        raise NotImplemented

    n_samples = X.shape[0]
    print(n_samples)

    if order == 'date':
        idx = np.arange(n_samples)

    elif order == 'random':
        idx = np.arange(n_samples)
        np.random.shuffle(idx)

    elif order == 'windspeed':
        idx = np.argsort(Y[:, 0])

    else:
        raise  NotImplemented

    X = X[idx, :]
    T = T[idx, :]
    Y = Y[idx, :]
    D = D[idx]
    W = W[idx]

    idx = np.arange(n_samples)
    tr_idx = idx[:int(n_samples * (1 - inf_split))]
    inf_idx = idx[int(n_samples * (1 - inf_split)):]

    if shuffle_train:
        np.random.shuffle(tr_idx)

    X_inf = X[inf_idx, :]
    T_inf = T[inf_idx, :]
    Y_inf = Y[inf_idx]
    D_inf = D[inf_idx]
    W_inf = W[inf_idx]

    X_tr = X[tr_idx, :]
    T_tr = T[tr_idx, :]
    Y_tr = Y[tr_idx]
    D_tr = D[tr_idx]
    W_tr = W[tr_idx]

    # split to train and validation parts

    idx = np.arange(X_tr.shape[0])
    tr_idx = idx[:int(X_tr.shape[0] * (1 - val_split))]
    val_idx = idx[int(X_tr.shape[0] * (1 - val_split)):]

    X_val = X_tr[val_idx, :]
    T_val = T_tr[val_idx, :]
    Y_val = Y_tr[val_idx]
    D_val = D_tr[val_idx]
    W_val = W_tr[val_idx]

    X_tr = X_tr[tr_idx, :]
    T_tr = T_tr[tr_idx, :]
    Y_tr = Y_tr[tr_idx]
    D_tr = D_tr[tr_idx]
    W_tr = W_tr[tr_idx]

    return X_tr, T_tr, Y_tr, D_tr, W_tr, X_val, T_val, Y_val, D_val, W_val, X_inf, T_inf, Y_inf, D_inf, W_inf

def sample_data(X, T, Y, D, W, sampling_rate=1.0):
    n = X.shape[0]
    idx = np.arange(0, n)
    np.random.shuffle(idx)
    idx = np.sort(idx[:int(sampling_rate * n)])
    X_ = X[idx, :]
    T_ = T[idx, :]
    Y_ = Y[idx]
    D_ = D[idx]
    W_ = W[idx]
    return X_, T_, Y_, D_, W_

def interpolate_negative_voltage(df, cols=['Current', 'Voltage']):
    df.loc[df['Voltage'] < 0, cols] = None
    df = df.interpolate()
    return df

def moving_average(df, window=60, id_col='ID', date_col='Date', current_col='Current', voltage_col='Voltage', windspeed_col='WindSpeed', lookback=100):
    ids = df[id_col].values
    dates = df[date_col].values
    features = df[[current_col, voltage_col, windspeed_col]].values
    unixtimes = dates.astype('datetime64[s]').astype('int')
    n_samples = df.shape[0]
    n_features = df.shape[1]
    ma_values = np.zeros((n_samples, n_features - 1))
    for i in range(n_samples):
        if i > 0 and i % 10000 == 0:
            print(f'{i / n_samples * 100:.1f} % completed')
        t = unixtimes[i]
        idx_lookback = np.arange(i - lookback, i + 1)
        u = unixtimes[idx_lookback]
        idx = np.where(u >  t - window)[0]
        ma_values[i, :] = np.mean(features[idx_lookback[idx], :], 0)
    return np.hstack([dates.reshape(-1, 1), ma_values])

if __name__ == '__main__':

    raw_fpath = osp.join(DATA_DIR, 'windside', 'raw')
    feature_fpath = osp.join(DATA_DIR, 'windside', 'features.csv')
    output_fpath = osp.join(DATA_DIR, 'windside', '{0}_{1}.{2}')

    extract_data(raw_fpath, feature_fpath, chunk_size=None)

    #feature_extractor = 'raw'
    #create_dataset(feature_fpath, output_fpath, feature_extractor=feature_extractor, chunk_size=1000)