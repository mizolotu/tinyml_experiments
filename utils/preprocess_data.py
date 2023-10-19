import mat73, json, scipy.io, os
import os.path as osp
import pandas as pd
import numpy as np
import tensorflow as tf

from ctypes import cdll, c_short, POINTER

def load_skab(fpath, n_samples=None, series_len=32):
    df = pd.read_csv(fpath, delimiter=';')
    keys = [item for item in df.keys()]
    label_key = 'anomaly'
    other_keys = ['changepoint', 'datetime']
    assert label_key in keys
    feature_keys = [key for key in keys if key not in other_keys and key != label_key]
    X = df[feature_keys].values
    Y = df[label_key].values.reshape(-1, 1)

    # series

    n_series = X.shape[0] - series_len + 1
    n_features = X.shape[1]
    X_ = np.zeros((n_series, n_features))
    for i in range(n_series):
        series = X[i : i + series_len, :]
        X_[i, :] = np.max(series, 0) # or np.mean?
    X = X_.copy()
    Y = Y[series_len - 1:, :]

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx, :]
    Y = Y[idx, :]
    if n_samples is not None:
        X = X[:n_samples, :]
        Y = Y[:n_samples, :]

    return {'X': np.array(X, dtype=float), 'Y': Y}

def load_odds(fpath, n_samples=None):
    try:
        mat = scipy.io.loadmat(fpath)
    except NotImplementedError:
        mat = mat73.loadmat(fpath)
    except:
        with open(fpath, 'r') as fp:
            lines = fp.readlines()
        lines = [line.strip() for line in lines]
        data = False
        categorical = []
        label_key = 'class'
        mat = {'X': [], 'y': []}
        for line in lines:
            if not data:
                if "@attribute" in line:
                    attri = line.split()
                    attr_name = attri[attri.index("@attribute") + 1]
                    value_set = ''.join(attri[attri.index("@attribute") + 2:])
                    if value_set.startswith('{') and value_set.endswith('}'):
                        value_set = sorted(value_set.split('{')[1].split('}')[0].split(','))
                        if attr_name == label_key:
                            assert len(value_set) == 2
                            label_position = len(categorical)
                        categorical.append(value_set)
                    else:
                        categorical.append(None)
                elif "@data" in line:
                    data = True
            else:
                x = []
                for i, (item, c) in enumerate(zip(line.split(','), categorical)):
                    if i == label_position:
                        y = c.index(item)
                    elif c is not None:
                        x_ = np.zeros(len(c))
                        x_[c.index(item)] = 1
                        x.append(x_)
                    else:
                        x.append(float(item))
                mat['X'].append(np.hstack(x))
                mat['y'].append(y)
        mat['X'] = np.vstack(mat['X'])
        mat['y'] = np.hstack(mat['y']).reshape(-1, 1)

    X = mat['X']
    Y = mat['y']

    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx, :]
    Y = Y[idx, :]

    if n_samples is not None:
        X = X[:n_samples, :]
        Y = Y[:n_samples, :]

    return {'X': np.array(X, dtype=float), 'Y': Y}

def raw(X):
    return X

def pam(X):
    assert len(X.shape) == 3
    m = X.shape[2]
    I = np.ones((m, m))
    I[np.triu_indices(m)] = 0
    E = np.vstack([
        np.hstack([
            np.min(x, 0),
            np.max(x, 0),
            np.mean(x, 0),
            np.std(x, 0),
        ]) for x in X
    ])
    return E

def freq1(X):
    print(X[0, :])
    assert len(X.shape) == 3
    return np.vstack([x[0, :] for x in X])

def flat(X):
    assert len(X.shape) == 3
    return np.vstack([x.flatten() for x in X])

def interval_fix_fft(x, step=8, m=3, n_fft_features=1, fpath='libraries/fix_fft_32k_dll/fix_fft_32k.so'):
    ff = cdll.LoadLibrary(fpath)
    ff.fix_fft.argtypes = [POINTER(c_short), POINTER(c_short), c_short, c_short]
    nsteps = len(x) // step
    w = tf.cast(x, tf.int32)
    intervals = np.split(w, nsteps)
    def fix_fft(re):
        s = np.zeros((n_fft_features, re.shape[1]))
        for j in range(re.shape[1]):
            im = [0 for _ in range(step)]
            re_c = (c_short * step)(*re[:, j])
            im_c = (c_short * step)(*im)
            ff.fix_fft(re_c, im_c, c_short(m), c_short(0))
            for i in range(n_fft_features):
                s[i, j] = np.round(np.sqrt(re_c[i] * re_c[i] + im_c[i] * im_c[i]) // 2)
        return s
    mgn = map(fix_fft, intervals)
    return np.vstack(mgn)

def fix_fft(x, m=5, n_fft_features=16, fpath='libraries/fix_fft_32k_dll/fix_fft_32k.so'):
    ff = cdll.LoadLibrary(fpath)
    ff.fix_fft.argtypes = [POINTER(c_short), POINTER(c_short), c_short, c_short]
    n = x.shape[0]
    x = [np.array(x[:, i], dtype=int) for i in range(x.shape[1])]
    def fft(re):
        im = [0 for _ in range(n)]
        re_c = (c_short * n)(*re)
        im_c = (c_short * n)(*im)
        ff.fix_fft(re_c, im_c, c_short(m), c_short(0))
        s = np.zeros(n_fft_features)
        for i in range(n_fft_features):
            s[i] = np.round(np.sqrt(re_c[i] * re_c[i] + im_c[i] * im_c[i]) // 2)
        return s
    mgn = map(fft, x)
    return np.transpose(np.vstack(mgn))

def fft(X, xmin=-8192, xmax=8192):
    assert len(X.shape) == 3
    x_min = np.min(X)
    x_max = np.max(X)
    X = (X - x_min) / (x_max - x_min + 1e-10)
    X = X * (xmax - xmin) + xmin
    X = np.round(X)
    X = np.clip(X, xmin, xmax)
    X = [np.round(x - np.mean(x, 0)) for x in X]
    E = [fix_fft(x) for x in X]
    print(E[0])
    #E = [interval_fix_fft(x) for x in X]
    return np.stack(E)


def load_dataset(data_dir, series_len, series_step, labels, feature_extractors=[], seed=0, check_baseline=False, n_samples=None):
    np.random.seed(seed)
    sample_subdirs = [subdir for subdir in os.listdir(data_dir) if osp.isdir(osp.join(data_dir, subdir))]
    X, Y = [], []
    for label_key in labels:
        for label_val in labels[label_key]:
            if label_val in sample_subdirs:
                sample_files = os.listdir(osp.join(data_dir, label_val))

                if check_baseline and 'baseline.csv' in sample_files:
                    fpath = osp.join(osp.join(data_dir, label_val), 'baseline.csv')
                    x = pd.read_csv(fpath, header=None, dtype=np.float).values
                    b_mean = np.mean(x, 0)
                    subtract_mean = True
                else:
                    subtract_mean = False

                for sf in sample_files:
                    fpath = osp.join(osp.join(data_dir, label_val), sf)
                    if osp.isfile(fpath) and fpath.endswith('.csv'):
                        x = pd.read_csv(fpath, header=None, dtype=np.float).values
                        if subtract_mean:
                            x -= b_mean[None, :]
                        n = x.shape[0]
                        y = np.ones((n, 1)) * label_key
                        for j in range(0, n - series_len, series_step):
                            s_x = x[j : j + series_len, :]
                            s_y = y[j + series_len, 0]
                            X.append(s_x)
                            Y.append(s_y)
    X = np.array(X)
    Y = np.vstack(Y)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx, :]
    Y = Y[idx, :]

    if n_samples is not None:
        print(f'{n_samples} out of {X.shape[0]} samples will be used')
        X = X[:n_samples, :]
        Y = Y[:n_samples, :]


    for feature_extractor in feature_extractors:
        extract_features = globals()[feature_extractor]
        X = extract_features(X)

    return {'X': X, 'Y': Y}

def split_data(dataset, inf_split=0.3, val_split=0.3, train_on_anomalies=False, validate_on_anomalies=False, shuffle_features=True, seed=0):

    np.random.seed(seed)

    n_features = np.prod(dataset['X'].shape[1:])
    idx0 = np.where(dataset['Y'] == 0)[0]
    np.random.shuffle(idx0)

    split = {}
    split['inf'], train_val = np.split(idx0, [int(inf_split * len(idx0))])
    split['val'], split['tr'] = np.split(train_val, [int(val_split * len(train_val))])

    idx1 = np.where(dataset['Y'] > 0)[0]

    split1 = {}
    split1['inf'], train_val = np.split(idx1, [int(inf_split * len(idx1))])
    split1['val'], split1['tr'] = np.split(train_val, [int(val_split * len(train_val))])

    include1 = {
        'tr': train_on_anomalies,
        'val': validate_on_anomalies,
        'inf': True
    }

    for key in split.keys():
        if include1[key]:
            idx1_key = np.random.choice(idx1, len(split[key]), replace=True)  # we use split[key] here instead of split1[key] because we want to equalize the numbers of normal and anomalous samples
            split[key] = np.append(split[key], idx1_key)

    idx = np.arange(dataset['X'].shape[1])
    if shuffle_features:
        np.random.shuffle(idx)
        idx = idx[:n_features]

    X = dataset['X'][:, idx]
    Y = dataset['Y'].squeeze()
    data = {}

    for key in split.keys():
        idx = split[key]
        np.random.shuffle(idx)
        data[key] = [X[split[key], :], Y[split[key]]]

    return data


if __name__ == '__main__':

    data_dir = 'data/benchmarks'
    dataset_files = os.listdir(data_dir)

    with open(f'{data_dir}/metainfo.json', 'r') as fp:
        metainfo = json.load(fp)

    for dataset in metainfo:
        fpath = osp.join(data_dir, dataset['file'])
        loader = globals()[f"load_{dataset['benchmark']}"]
        X, Y = loader(fpath)
        print(fpath, X.shape, Y.shape, len(np.unique(Y)))




