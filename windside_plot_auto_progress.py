import argparse as arp
import json
import os.path as osp

import numpy as np
import pandas as pd

from config import WINDSIDE_MODEL_DIR, WINDSIDE_FIG_DIR
from matplotlib import pyplot as pp
from matplotlib.colors import BASE_COLORS, CSS4_COLORS
from matplotlib.markers import MarkerStyle as ms

from utils.boards import get_ram_flash


def moving_average(x, step=1, window=10):
    n_samples = len(x)
    y = np.zeros(n_samples)
    for i in range(n_samples):
        x_ = x[np.maximum(0, i-window+1):i+1]
        y[i] = np.mean(x_)
    return y

def moving_minimum(x):
    n_samples = len(x)
    y = np.zeros(n_samples)
    for i in range(n_samples):
        x_ = x[0 : i + 1]
        y[i] = np.min(x_)
    return y

if __name__ == '__main__':
    parser = arp.ArgumentParser(description='Plot progress of the auto regression model to estimate wind speed.')
    parser.add_argument('-o', '--order', help='Order', default='date', choices=['random', 'date', 'windspeed'])
    parser.add_argument('-l', '--label_mode', help='Label mode', default='avg+max', choices=['avg', 'max', 'avg+max'])
    parser.add_argument('-b', '--boards', help='Board names', nargs='+', default=['lora'])
    parser.add_argument('-a', '--algorithms', help='Algorithms', default=['dnn', 'cnn'], choices=['gbdt', 'cnn', 'dnn'], nargs='+')
    parser.add_argument('-f', '--frameworks', help='Frameworks', nargs='+', default=['ray', 'flaml'], choices=['ray', 'flaml']) # 'ray', 'flaml'
    args = parser.parse_args()

    error_key = 'error1'
    error_idx = 1
    col_idx_to_print = error_idx
    fig_sizes = [
        # (5, 2.5),
        (7, 4.3),
        # (10, 5.0)
    ]

    baseline_model_fname = f'baseline_lr'
    baseline_model_fpath = osp.join(WINDSIDE_MODEL_DIR, baseline_model_fname)
    print('Baseline model fpath:', baseline_model_fpath)
    with open(osp.join(baseline_model_fpath, 'stats.json')) as f:
        baseline_stats = json.load(f)
        baseline_error = baseline_stats[error_key][error_idx]

    x = None

    pp.style.use('default')
    colormap = 'Dark2'

    # colors = ['orangered', 'teal', 'darkmagenta', 'midnightblue']
    colors = ['black'] + [c for c in BASE_COLORS.values()][0:6] + [
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

    idx = [3, 4]

    colors = [colors[i] for i in idx]
    markers = [markers[i] for i in idx]

    line_styles = ['-', '--']

    algorithm_names = {
        'gbdt': 'Gradient boosting decision trees',
        'dnn': 'Fully-connected neural network',
        'cnn': 'Convolutional neural network',
    }

    board_names = {
        'sense': 'Arduino Nano 33 Sense',
        'iot': 'Arduino Nano 33 Sense',
        'every': 'Arduino Nano Every',
        'uno': 'Arduino Uno',
        'lora': 'LoRa E5 mini',
    }

    best_vals = np.zeros((len(args.boards), len(args.frameworks), len(args.algorithms)))
    best_hps = np.zeros((len(args.boards), len(args.frameworks), len(args.algorithms)), dtype='object')

    n_points = 100
    n_markers = 32

    for bi, board in enumerate(args.boards):

        Y, L = [], []

        for fi, framework in enumerate(args.frameworks):

            for ai, alg in enumerate(args.algorithms):

                model_fname = f'{board}_{alg}_{args.order}_{args.label_mode}_{framework}'
                model_fpath = osp.join(WINDSIDE_MODEL_DIR, model_fname)
                progress_fpath = osp.join(model_fpath, 'progress.csv')
                print('Progress fpath:', progress_fpath)

                vals = pd.read_csv(progress_fpath, header=None).values
                v = vals[:, col_idx_to_print]

                print(board, alg, vals[np.argmin(v)])

                #y = moving_average(v)
                y = moving_minimum(v)

                best_idx = np.argmin(v)
                print(f'Best results for {alg} on {board}:', vals[best_idx, :])

                best_vals[bi, fi, ai] = v[best_idx]
                best_hps[bi, fi, ai] = ', '.join([str(item) for item in vals[best_idx, 3:]])

                n_samples = np.minimum(len(y), n_points)
                if x is None:
                    x = np.arange(n_samples) + 1
                else:
                    assert len(x) == n_samples

                Y.append(y)
                L.append(f'{framework.capitalize()}+{algorithm_names[alg]}')

        y_b = np.ones(len(x)) * baseline_error

        #Y = [y_b] + Y
        #L = ['Baseline (linear regression)'] + L

        for fsi, fig_size in enumerate(fig_sizes):
            fig, ax = pp.subplots()
            fig.set_size_inches(*fig_size)

            for fi, framework in enumerate(args.frameworks):

                for ai, alg in enumerate(args.algorithms):
                    i = fi * len(args.algorithms) + ai

                    idx = np.arange(0, n_points, int(n_points / n_markers))
                    idx = np.hstack([idx, n_points - 1])

                    pp.plot(x[idx], Y[i][idx], f'{markers[ai]}{line_styles[fi]}', color=colors[ai], linewidth=1.5, label=L[i])
                    # pp.plot(x, Y[i], f'{line_styles[fi]}', color=colors[ai], linewidth=2, label=L[i])

            ram, flash = get_ram_flash(board=board)
            title = f'{board_names[board]} ({ram} KB RAM)'
            # pp.title(title)

            pp.xlim([1, x[-1]])
            pp.xlabel('Iteration')
            if col_idx_to_print == 0:
                pp.ylabel('Avg+max wind speed MAE, m/s')
                fname_suffix = 'avg+max'
            elif col_idx_to_print == 1:
                pp.ylabel('Average wind speed MAE, m/s')
                fname_suffix = 'avg'
            elif col_idx_to_print == 2:
                pp.ylabel('Maximal wind speed MAE, m/s')
                fname_suffix = 'max'
            pp.legend()
            # if fsi == 0:
            #    size = "small"
            # elif fsi == 1:
            #    size = "medium"
            # else:
            #    size = "big"
            for ext in ['png', 'pdf']:
                # pp.savefig(osp.join(ARE_FIG_DIR, f'{board}_auto_{",".join(args.frameworks)}_progress_{size}.{ext}'), bbox_inches='tight', pad_inches=0)
                #pp.savefig(osp.join(WINDSIDE_FIG_DIR, f'{board}_{",".join(args.frameworks)}_progress.{ext}'), bbox_inches='tight', pad_inches=0)
                pp.savefig(osp.join(WINDSIDE_FIG_DIR, f'{board}_{args.order}_{fname_suffix}_{",".join(args.frameworks)}_progress.{ext}'), bbox_inches='tight', pad_inches=0)
            pp.close()

    best_hps_lines = []
    for fi, framework in enumerate(args.frameworks):
        for ai, alg in enumerate(args.algorithms):
            row = [f'{framework.upper()}+{alg.upper()}']
            for i, board in enumerate(args.boards):
                if best_vals[i, fi, ai] == np.min(best_vals[i, :]):
                    row.append(f'\\textbf{{{best_vals[i, fi, ai]:.4f}}}')
                    best_hps_lines.append(f'{board} {framework} {alg} {best_hps[i, fi, ai]}')
                else:
                    row.append(f'{best_vals[i, fi, ai]:.4f}')
            print(' & '.join(row))


        if 0:
            for i, (y, l) in enumerate(zip(Y, L)):
                pp.plot(x, y, color=colors[i], linewidth=2, label=l)

            ram, flash = get_ram_flash(board=board)
            title = f'{board_names[board]} ({flash} KB memory)'
            pp.title(title)

            pp.xlim([0, x[-1]])
            pp.xlabel('Iteration')
            if error_idx == 0:
                pp.ylabel('Average wind speed MAE, m/s')
            elif error_idx == 1:
                pp.ylabel('Maximal wind speed MAE, m/s')
            pp.legend()
            for ext in ['png', 'pdf']:
                pp.savefig(osp.join(WINDSIDE_FIG_DIR, f'{board}_{args.order}_{args.label_mode}_{args.framework}_progress.{ext}'), bbox_inches='tight', pad_inches=0)
            pp.close()

    for line in best_hps_lines:
        print(line)