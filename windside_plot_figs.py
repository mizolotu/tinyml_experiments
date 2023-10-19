import json
import os
import os.path as osp
import pandas as pd
import numpy as np

from config import WINDSIDE_DATA_DIR, FIG_DIR, WINDSIDE_MODEL_DIR
from matplotlib import pyplot as pp
from time import time
from datetime import datetime
from windside_extract_data import interpolate_negative_voltage, moving_average

from scipy.stats import binned_statistic

from matplotlib.colors import BASE_COLORS, CSS4_COLORS
from matplotlib.markers import MarkerStyle as ms

def plot_histogram(bin_step=0.025):

    input_fpath = osp.join(WINDSIDE_DATA_DIR, 'features.csv')
    if not osp.isdir(FIG_DIR):
        os.mkdir((FIG_DIR))
    if not osp.isdir(osp.join(FIG_DIR, 'windside')):
        os.mkdir(osp.join(FIG_DIR, 'windside'))

    chunksize = None

    if chunksize is None:
        df = pd.read_csv(input_fpath)
    else:
        df = pd.read_csv(input_fpath, chunksize=chunksize)
        df = next(df)

    vals = df.values

    current_col_idx = 2
    windspeed_col_idx = 4

    idx0 = np.where(vals[:, current_col_idx] == 0)[0]
    idx1 = np.where(vals[:, current_col_idx] > 0)[0]

    print(len(idx0), len(idx1))

    windspeed1 = vals[idx1, windspeed_col_idx]

    w_max = np.max(windspeed1)
    bins = np.arange(0, (w_max // bin_step + 1) * bin_step, bin_step)

    h, b = np.histogram(windspeed1, bins)

    fig, ax = pp.subplots(1)
    fig.set_size_inches(*fig_size)

    x = bins[1:]
    y = h

    x_axis = np.arange(len(x))

    pp.bar(x_axis, y, label='Wind speed histogram', color=colors[1])

    xticks_step = 2 / bin_step
    idx = np.arange(xticks_step, w_max / bin_step, xticks_step).astype(int) - 1

    pp.xticks(x_axis[idx], [int(x[i]) for i in idx])
    pp.xlabel("Wind speed, m/s")
    pp.ylabel("Number of samples")
    #pp.yscale('log')
    pp.xlim([x_axis[0], x_axis[-1]])

    pp.legend()

    for ext in ['png', 'pdf']:
        pp.savefig(osp.join(FIG_DIR, 'windside', f'windspeed_hist.{ext}'), bbox_inches='tight', pad_inches=0)


def plot_current_vs_windspeed():

    input_fpath = osp.join(WINDSIDE_DATA_DIR, 'features.csv')
    if not osp.isdir(FIG_DIR):
        os.mkdir((FIG_DIR))
    if not osp.isdir(osp.join(FIG_DIR, 'windside')):
        os.mkdir(osp.join(FIG_DIR, 'windside'))

    chunksize = None

    if chunksize is None:
        df = pd.read_csv(input_fpath)
    else:
        df = pd.read_csv(input_fpath, chunksize=chunksize)
        df = next(df)

    vals = df.values
    print(vals.shape, vals[0, 0], vals[-1, 0])

    current_col_idx = 2
    windspeed_col_idx = 4

    idx0 = np.where(vals[:, current_col_idx] == 0)[0]
    idx1 = np.where(vals[:, current_col_idx] > 0)[0]

    print(len(idx0), len(idx1))

    current1 = vals[idx1, current_col_idx]
    windspeed1 = vals[idx1, windspeed_col_idx]
    current = vals[:, current_col_idx]
    windspeed = vals[:, windspeed_col_idx]
    u_current = np.unique(current)

    t_start = time()

    x_bar = []
    y_bar_means, y_bar_mins, y_bar_maxs = [], [], []
    step = 1
    current_first = np.min(u_current)
    current_last = np.ceil(np.max(u_current))
    for c in np.arange(current_first, current_last + step, step):
        x_bar.append(c)
        if c == 0:
            y_bar_means.append(np.mean(windspeed[idx0]))
            y_bar_mins.append(np.min(windspeed[idx0]))
            y_bar_maxs.append(np.max(windspeed[idx0]))
        else:
            idx = np.where((current1 > c - step) & (current1 <= c))[0]
            y_bar_means.append(np.mean(windspeed1[idx]))
            y_bar_mins.append(np.min(windspeed1[idx]))
            y_bar_maxs.append(np.max(windspeed1[idx]))
    y_bar_means = np.array(y_bar_means)
    y_bar_mins = np.array(y_bar_mins)
    y_bar_maxs = np.array(y_bar_maxs)

    fig, ax = pp.subplots(1)
    fig.set_size_inches(*fig_size)
    ax.plot(x_bar, y_bar_means, 'b-', label='Average')
    pp.errorbar(x_bar, y_bar_means, [y_bar_means - y_bar_mins, y_bar_maxs - y_bar_means], fmt='ob', ecolor='k', lw=3, capsize=3, label='Average, minimal and maximal')
    pp.xlim([np.min(x_bar) - 0.1, np.max(x_bar) + 0.1])
    x_ticks = x_bar.copy()
    pp.xticks(x_ticks)
    pp.xlabel('Current, A')
    pp.ylabel('Wind speed, m/s')
    pp.legend()
    for ext in ['png', 'pdf']:
        pp.savefig(osp.join(FIG_DIR, 'windside', f'errorbar.{ext}'), bbox_inches='tight', pad_inches=0)
    pp.close()

    print(f'Error bars have been plotted in {time() - t_start} seconds!')

    t_start = time()

    x_bar = []
    y_bar_means, y_bar_mins, y_bar_maxs = [], [], []
    step = 0.01
    current_first = np.min(u_current)
    current_last = np.ceil(np.max(u_current))
    for c in np.arange(current_first, current_last + step, step):
        if c == 0:
            x_bar.append(c)
            y_bar_means.append(np.mean(windspeed[idx0]))
            y_bar_mins.append(np.min(windspeed[idx0]))
            y_bar_maxs.append(np.max(windspeed[idx0]))
        else:
            idx = np.where((current1 > c - step) & (current1 <= c))[0]
            if len(idx) > 0:
                x_bar.append(c)
                y_bar_means.append(np.mean(windspeed1[idx]))
                y_bar_mins.append(np.min(windspeed1[idx]))
                y_bar_maxs.append(np.max(windspeed1[idx]))
    y_bar_means = np.array(y_bar_means)
    y_bar_mins = np.array(y_bar_mins)
    y_bar_maxs = np.array(y_bar_maxs)

    fig, ax = pp.subplots(1)
    fig.set_size_inches(*fig_size)
    ax.plot(x_bar, y_bar_means, 'b-', label='Average wind speed')
    ax.fill_between(x_bar, y_bar_mins, y_bar_maxs, facecolor='b', alpha=0.3, label='Min/max wind speed')
    pp.xlim([np.min(x_bar) - 0.1, np.max(x_bar) + 0.1])
    x_ticks = np.arange(np.max(x_bar))
    pp.xticks(x_ticks)
    pp.xlabel('Current, A')
    pp.ylabel('Wind speed, m/s')
    pp.legend()
    for ext in ['png', 'pdf']:
        pp.savefig(osp.join(FIG_DIR, 'windside', f'fill_between.{ext}'), bbox_inches='tight', pad_inches=0)
    pp.close()

    print(f'Lines have been plotted in {time() - t_start} seconds!')

    t_start = time()
    print(len(u_current))
    n = len(np.unique(current))
    x = np.array(u_current, dtype=float)
    y = np.zeros((n, 3))
    for i in range(n):
        if i == 0:
            y[i, 0] = np.mean(windspeed[idx0])
            y[i, 1] = np.min(windspeed[idx0])
            y[i, 2] = np.max(windspeed[idx0])
        else:
            idx = np.where(current1 == u_current[i])[0]
            y[i, 0] = np.mean(windspeed1[idx])
            y[i, 1] = np.min(windspeed1[idx])
            y[i, 2] = np.max(windspeed1[idx])

    fig, ax = pp.subplots(1)
    fig.set_size_inches(*fig_size)
    cmap = pp.get_cmap("cool")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    ax.scatter(x, y[:, 0], linewidths=0.01, edgecolors='grey',  c=cmap(rescale(y[:, 0])), label='Data sample')
    pp.xlim([np.min(x) - 0.1, np.max(x) + 0.1])
    pp.xticks(x_ticks)
    pp.xlabel('Current, A')
    pp.ylabel('Wind speed, m/s')
    pp.legend()
    for ext in ['png', 'pdf']:
        pp.savefig(osp.join(FIG_DIR, 'windside', f'scatter.{ext}'), bbox_inches='tight', pad_inches=0)

    print(f'Scatter has been plotted in {time() - t_start} seconds!')

def plot_model_bars_():
    model_dir = 'windside_models'
    model_dpaths = [osp.join(model_dir, item) for item in os.listdir(model_dir) if osp.isdir(osp.join(model_dir, item))]

    x = [
        'Baseline,\nmean values',
        ''
        #'Only time,\nDense',
        #'Time & current,\nDense',
        #'Only current,\nDense',
        #'Only current,\nConv1D',
        #'Only current,\nLSTM'
    ]

    y = np.zeros((len(x), 3))

    for dpath in model_dpaths:
        with open(osp.join(dpath, 'stats.json')) as f:
            stats = json.load(f)

        vals = np.array([stats['error0'], stats['error1'], stats['error']])
        arch = stats['model']

        if dpath.endswith('baseline'):
            y[0, :] = vals
        elif '\"t\"' in arch and '\"x\"' not in arch:
            y[1, :] = vals
        elif '\"t\"' in arch and '\"x\"' in arch:
            y[2, :] = vals
        elif '\"t\"' not in arch and '\"x\"' in arch and 'conv' in arch:
            y[4, :] = vals
        elif '\"t\"' not in arch and '\"x\"' in arch and 'lstm' in arch:
            y[5, :] = vals
        elif '\"t\"' not in arch and '\"x\"' in arch and 'dense' in arch:
            y[3, :] = vals

        print(vals)

        fig, ax = pp.subplots(1)
        fig.set_size_inches(9, 4)

        x_axis = np.arange(len(x))
        pp.bar(x_axis - 0.2, y[:, 0], 0.2, label='Current = 0', color='crimson')
        pp.bar(x_axis, y[:, 1], 0.2, label='Current > 0', color='green')
        pp.bar(x_axis + 0.2, y[:, 2], 0.2, label='All samples', color='midnightblue')

        pp.xticks(x_axis, x)
        #pp.xlabel("Models")
        pp.ylabel("Mean absolute error")
        pp.legend()

        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(FIG_DIR, 'windside', f'model_bars.{ext}'), bbox_inches='tight', pad_inches=0)

def plot_linear_model_bars():
    model_dir = WINDSIDE_MODEL_DIR
    model_dpaths = [osp.join(model_dir, item) for item in os.listdir(model_dir) if osp.isdir(osp.join(model_dir, item)) and ('filter-zeros_mean' in item or 'baseline' in item)]

    model_dict = {
        #"{}": 'Baseline,\nmean values',
        "\"LR\"": 'Linear\nregression',
        "\"ENR\"": 'Elastic net\nregression',
        "\"SGDR\"": 'SGD\nregression',
        "\"BRR\"": 'Bayesian ridge\nregression',
        "\"SVR\"": 'Support vector \nregression'
    }

    model_dict_keys = [key for key in model_dict.keys()]

    y = np.zeros((len(model_dict_keys), 3))

    x = ['' for _ in range(len(model_dict_keys))]

    for dpath in model_dpaths:

        try:
            with open(osp.join(dpath, 'stats.json')) as f:
                stats = json.load(f)
        except:
            continue

        vals = np.array([stats['error0'], stats['error1'], stats['error']])
        #print(dpath, stats)
        try:
            model_name = stats['model']
        except:
            model_name = None

        if model_name is not None and model_name in model_dict_keys:
            idx = model_dict_keys.index(model_name)
            x[idx] = model_dict[model_name]
            y[idx, :] = vals

            print(dpath, model_name, vals)

    fig, ax = pp.subplots(1)
    fig.set_size_inches(*fig_size)

    x_axis = np.arange(len(x))
    pp.bar(x_axis - 0.2, y[:, 0], 0.2, label='Current = 0', color='crimson')
    pp.bar(x_axis, y[:, 1], 0.2, label='Current > 0', color='green')
    pp.bar(x_axis + 0.2, y[:, 2], 0.2, label='All samples', color='midnightblue')

    pp.xticks(x_axis, x)
    #pp.xlabel("Models")
    pp.ylabel("Mean absolute error")
    pp.legend()

    for ext in ['png', 'pdf']:
        pp.savefig(osp.join(FIG_DIR, 'windside', f'linear_model_bars.{ext}'), bbox_inches='tight', pad_inches=0)

def plot_tree_model_bars():
    model_dir = WINDSIDE_MODEL_DIR
    model_dpaths = [osp.join(model_dir, item) for item in os.listdir(model_dir) if osp.isdir(osp.join(model_dir, item)) and ('filter-zeros_mean' in item)]

    model_dict = {
        "\"LR\"": 'Linear\nregression',
        #"\"DTR\"": 'Decision tree\nregression',
        #"\"RFR\"": 'Random forest\nregression',
        "\"XGBRFR\"": 'Random forest\nregression',
        "\"GBR\"": 'Sklearn-GBDT\nregression',
        "\"XGBR\"": 'XGBoost-GBDT\nregression',
        "\"LGBMR\"": 'LightGBM-GBDT\nregression'
    }

    model_dict_keys = [key for key in model_dict.keys()]

    y = np.zeros((len(model_dict_keys), 3))

    x = ['' for _ in range(len(model_dict_keys))]

    print('\n')

    for dpath in model_dpaths:

        try:
            with open(osp.join(dpath, 'stats.json')) as f:
                stats = json.load(f)
        except:
            continue

        vals = np.array([stats['error0'], stats['error1'], stats['error']])

        try:
            model_name = stats['model']
        except:
            model_name = None

        if model_name is not None and model_name in model_dict_keys:
            idx = model_dict_keys.index(model_name)
            x[idx] = model_dict[model_name]
            y[idx, :] = vals

            print(model_name, vals)

    fig, ax = pp.subplots(1)
    fig.set_size_inches(*fig_size)

    x_axis = np.arange(len(x))
    pp.bar(x_axis - 0.2, y[:, 0], 0.2, label='Current = 0', color='crimson')
    pp.bar(x_axis, y[:, 1], 0.2, label='Current > 0', color='green')
    pp.bar(x_axis + 0.2, y[:, 2], 0.2, label='All samples', color='midnightblue')

    pp.xticks(x_axis, x)
    #pp.xlabel("Models")
    pp.ylabel("Mean absolute error")
    #pp.legend()

    for ext in ['png', 'pdf']:
        pp.savefig(osp.join(FIG_DIR, 'windside', f'tree_model_bars.{ext}'), bbox_inches='tight', pad_inches=0)

def plot_deep_model_bars():
    model_dir = WINDSIDE_MODEL_DIR
    model_dpaths = [osp.join(model_dir, item) for item in os.listdir(model_dir) if osp.isdir(osp.join(model_dir, item)) and ('filter-zeros_mean' in item)]

    model_dict = {
        "\"LR\"": 'Linear\nregression',
        "{\"x\": [\"dense_64\", \"dense_64\"], \"c\": []}": 'Deep learning\nFC_64->FC_64',
        "{\"x\": [\"dense_64\", \"dense_64\", \"dense_64\"], \"c\": []}": 'Deep learning\nFC_64->FC_64->FC_64',
        "{\"x\": [\"conv_16\", \"dense_64\"], \"c\": []}": 'Deep learning\nConv_16->FC_64',
        "{\"x\": [\"lstm_64\"], \"c\": []}": 'Deep learning\nLSTM_64',
    }

    model_dict_keys = [key for key in model_dict.keys()]

    y = np.zeros((len(model_dict_keys), 3))

    x = ['' for _ in range(len(model_dict_keys))]

    print('\n')

    for dpath in model_dpaths:

        try:
            with open(osp.join(dpath, 'stats.json')) as f:
                stats = json.load(f)
        except:
            continue

        vals = np.array([stats['error0'], stats['error1'], stats['error']])
        try:
            model_name = stats['model']
        except:
            model_name = None

        if model_name is not None and model_name in model_dict_keys:
            idx = model_dict_keys.index(model_name)
            x[idx] = model_dict[model_name]
            y[idx, :] = vals

            print(model_name, vals)

    fig, ax = pp.subplots(1)
    fig.set_size_inches(*fig_size)

    x_axis = np.arange(len(x))
    pp.bar(x_axis - 0.2, y[:, 0], 0.2, label='Current = 0', color='crimson')
    pp.bar(x_axis, y[:, 1], 0.2, label='Current > 0', color='green')
    pp.bar(x_axis + 0.2, y[:, 2], 0.2, label='All samples', color='midnightblue')

    pp.xticks(x_axis, x)
    #pp.xlabel("Models")
    pp.ylabel("Mean absolute error")
    #pp.legend()

    for ext in ['png', 'pdf']:
        pp.savefig(osp.join(FIG_DIR, 'windside', f'deep_model_bars.{ext}'), bbox_inches='tight', pad_inches=0)

def plot_windspeed_vs_power(interpolate_negative_voltage=False, skip_negative_voltage=True):

    input_fpath = osp.join(WINDSIDE_DATA_DIR, 'features.csv')

    if not osp.isdir(FIG_DIR):
        os.mkdir((FIG_DIR))
    if not osp.isdir(osp.join(FIG_DIR, 'windside')):
        os.mkdir(osp.join(FIG_DIR, 'windside'))

    #date_col = 'Date'
    #date_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    chunksize = None

    if chunksize is None:
        df = pd.read_csv(
            input_fpath,
            #parse_dates=[date_col], date_parser=date_parser
        )
    else:
        df = pd.read_csv(
            input_fpath, chunksize=chunksize,
            #parse_dates=[date_col], date_parser=date_parser
        )
        df = next(df)

    print(df.keys())

    print(df.values[53000:53004, :])

    if interpolate_negative_voltage:
        df = interpolate_negative_voltage(df, cols=['Current', 'Voltage', 'WindSpeed'])

    print(df.values[53000:53004, :])

    vals = df.values
    #vals = moving_average(df)

    print(vals.shape, vals[0, 0], vals[-1, 0])

    voltage = vals[:, 2]

    print(len(np.where(voltage > 0)[0]), len(np.where(voltage < 0)[0]))

    if skip_negative_voltage:
        idx = np.where(voltage >= 0)[0]
    else:
        idx = np.arange(vals.shape[0])

    voltage = voltage[idx]
    current = vals[idx, 1]
    power = current * np.abs(voltage)
    windspeed = vals[idx, 3]

    print(len(np.where(voltage > 0)[0]), len(np.where(voltage == 0)[0]), len(np.where(voltage < 0)[0]))
    print(len(np.where(current > 0)[0]), len(np.where(current == 0)[0]), len(np.where(current < 0)[0]))
    print(len(np.where(windspeed > 0)[0]), len(np.where(windspeed == 0)[0]), len(np.where(windspeed < 0)[0]))

    u_windspeed = np.unique(windspeed)

    t_start = time()

    x_bar = []
    y_bar_means, y_bar_mins, y_bar_maxs = [], [], []
    step = 0.5
    windspeed_first = np.floor(np.min(u_windspeed))
    windspeed_last = np.ceil(np.max(u_windspeed))
    for w in np.arange(windspeed_first, windspeed_last + step, step):
        idx = np.where((windspeed >= w) & (windspeed <= w + step))[0]
        if len(idx) > 0:
            x_bar.append(w + step)
            power_mean = np.mean(power[idx])
            power_min = np.min(power[idx])
            power_max = np.max(power[idx])
            y_bar_means.append(power_mean)
            y_bar_mins.append(power_min)
            y_bar_maxs.append(power_max)
            print(f'Wind speed range = [{w} m/s - {w + step} m/s]: power range = [{power_min} W - {power_max} W], power mean = {power_mean} W')

    y_bar_means = np.array(y_bar_means)
    y_bar_mins = np.array(y_bar_mins)
    y_bar_maxs = np.array(y_bar_maxs)

    fig, ax = pp.subplots(1)
    fig.set_size_inches(*fig_size)
    ax.plot(x_bar, y_bar_means, 'b-', label='Average')
    pp.errorbar(x_bar, y_bar_means, [y_bar_means - y_bar_mins, y_bar_maxs - y_bar_means], fmt='ob', ecolor='k', lw=3, capsize=3, label='Average, minimal and maximal')
    pp.xlim([np.min(x_bar) - 0.1, np.max(x_bar) + 0.1])
    x_ticks = x_bar.copy()
    pp.xticks(x_ticks)
    pp.xlabel('Wind speed, m/s')
    pp.ylabel('Power, W')
    pp.legend()
    for ext in ['png', 'pdf']:
        pp.savefig(osp.join(FIG_DIR, 'windside', f'power_per_windspeed.{ext}'), bbox_inches='tight', pad_inches=0)
    pp.close()

    print(f'Windspeed vs power error bars have been plotted in {time() - t_start} seconds!')

def plot_error_curves(bin_step=0.025):

    keys = ['Average', 'Maximum']

    models = {
        'Linear regression': 'lr_aec4c435-0411-33e9-ab2d-ea1f6e2e72de_date_avg+max',
        'Random forest': 'xgbrfr_d59578c3-79da-3d34-800f-7707bd1e1cda_date_avg+max',
        'Gradient boosted decision trees': 'xgbr_4605db49-7971-36eb-8fa2-625568190677_date_avg+max',
        'Fully-connected network': 'tf_fd2626b2-4e2a-3808-b1cd-cf17031a2017_date_avg+max',
        #'Fully-connected network (optimal hyperparams)': 'tf_d0a474c6-2dbe-35e0-a871-f98451f0b8d6_date_avg+max',
        #'Fully-connected network': 'sense_dnn_date_avg+max_flaml',
        'Convolutional network': 'tf_67d12edf-c094-3929-9d11-27615cdde5ba_date_avg+max',
        #'Convolutional network': 'sense_cnn_date_avg+max_flaml',
        'LSTM network': 'tf_60ac4c2c-c154-32ad-abe2-03755019051f_date_avg+max'
    }

    for ki, key in enumerate(keys):

        fig, ax = pp.subplots(1)
        fig.set_size_inches(*fig_size)

        n_bins, x = None, None

        models_dir = osp.join(WINDSIDE_MODEL_DIR)

        totals = []

        for ii, item in enumerate(models.keys()):
            print(item)
            for fname in os.listdir(models_dir):
                if fname == models[item]:
                    break
            fpath = osp.join(models_dir, fname, 'error.csv')
            vals = pd.read_csv(fpath, header=None).values

            if n_bins is None:
                n_bins = (np.max(vals[:, 0]) // bin_step) + 1

            if vals.shape[1] == 4:
                h, e, _ = binned_statistic(vals[:, ki * len(keys)], vals[:, ki * len(keys) + 1], bins=n_bins)
                if x is None:
                    x = vals[:, ki * len(keys)]
            else:
                h, e, _ = binned_statistic(x, vals[:, ki * len(keys)], bins=n_bins)
            # ax.plo1t(X_mean[idx_sorted], E[idx_sorted], M, label=L)

            totals.append(np.mean(vals[:, ki * len(keys) + 1]))

            idx_not_nan = np.where(np.isnan(h) == False)[0]
            x_ = e[1:][idx_not_nan]
            y_ = h[idx_not_nan]
            ax.plot(x_, y_, f'{markers[ii]}-', color=colors[ii], label=item)

        pp.xlim([np.min(x), np.max(x)])
        pp.xlabel(f'{key} wind speed, m/s')
        pp.ylabel('Mean absolute error, m/s')
        pp.legend()
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(FIG_DIR, 'windside', f'error_{key.lower()}.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

        fig, ax = pp.subplots(1)
        fig.set_size_inches(*fig_size)

        x_axis = np.arange(len(totals))
        pp.bar(x_axis, totals, color=colors)

        handles = [pp.Rectangle((0, 0), 1, 1, color=colors[ii]) for ii in range(len(models))]
        pp.legend(handles, models.keys(), bbox_to_anchor=(1.04, 1), loc="upper left")

        pp.xticks([])
        pp.ylabel('Mean absolute error, m/s')

        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(FIG_DIR, 'windside', f'error_{key.lower()}_total.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

if __name__ == '__main__':

    fig_size = (7, 4.3)
    small_fig_size = (6, 3.5)
    big_fig_size = (16, 11)

    colors = ['black'] + [c for c in BASE_COLORS.values()][:6] + [
        CSS4_COLORS['orangered'],
        CSS4_COLORS['sienna'],
        CSS4_COLORS['crimson'],
        CSS4_COLORS['dimgrey'],
        CSS4_COLORS['gold'],
        CSS4_COLORS['lawngreen'],
        CSS4_COLORS['deepskyblue'],
        CSS4_COLORS['midnightblue'],
        CSS4_COLORS['darkmagenta'],
    ]

    markers = list(ms.filled_markers)[:5] + ['s', '*']

    #plot_linear_model_bars()
    #plot_tree_model_bars()
    #plot_deep_model_bars()

    plot_histogram()

    plot_current_vs_windspeed()
    #plot_windspeed_vs_power()

    #plot_error_curves()