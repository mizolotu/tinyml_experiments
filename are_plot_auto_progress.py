import argparse as arp
import json
import os.path as osp

import numpy as np
import pandas as pd

from config import ARE_MODEL_DIR, ARE_FIG_DIR
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

def moving_maximum(x):
    n_samples = len(x)
    y = np.zeros(n_samples)
    for i in range(n_samples):
        x_ = x[0 : i + 1]
        y[i] = np.max(x_)
    return y

if __name__ == '__main__':
    parser = arp.ArgumentParser(description='Plot progress of the auto regression model to estimate wind speed.')
    parser.add_argument('-l', '--label', help='Label', default='mean', choices=['last', 'mean'])
    parser.add_argument('-z', '--zeros', help='Filter zeros?', default='filter', choices=['filter', 'only'])
    parser.add_argument('-b', '--boards', help='Board names', nargs='+', default=['uno', 'every', 'iot', 'sense', 'lora'])
    parser.add_argument('-a', '--algorithms', help='Algorithms', nargs='+', default=['dsvdd']) # 'cs', 'skm', 'dsvdd'
    parser.add_argument('-f', '--frameworks', help='Frameworks', nargs='+', default=['ray'], choices=['ray', 'flaml']) # 'ray', 'flaml'
    args = parser.parse_args()

    col_idx_to_print = 0
    error_key = 'acc'
    fig_sizes = [
        #(5, 2.5),
        (7, 4.3),
        #(10, 5.0)
    ]

    n_markers = 25

    baseline_model_fname = f'baseline_cluster1'
    baseline_model_fpath = osp.join(ARE_MODEL_DIR, baseline_model_fname)
    print('Baseline model fpath:', baseline_model_fpath)
    with open(osp.join(baseline_model_fpath, 'stats.json')) as f:
        baseline_stats = json.load(f)
        baseline_error = baseline_stats[error_key]

    x = None

    pp.style.use('default')
    colormap = 'Dark2'

    # colors = ['orangered', 'teal', 'darkmagenta', 'midnightblue']
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

    idx = [0, 1, 5]

    colors = [colors[i] for i in idx]
    markers = [markers[i] for i in idx]

    line_styles = ['-', '--']

    algorithm_names = {
        'skm': 'SKM', # 'Scalable k-means++',
        'cs': 'CS', # 'CluStrem',
        #'som': 'Self-organising map',
        'dsvdd': 'DSVDD' # 'Deep support vector data description'
    }


    board_names = {
        'uno': 'Arduino Uno Rev3',
        'every': 'Arduino Nano Every',
        'iot': 'Arduino Nano 33 IoT',
        'sense': 'Arduino Nano 33 BLE Sense',
        'lora': 'Wio-E5 mini'
    }

    best_vals = np.zeros((len(args.boards), len(args.frameworks), len(args.algorithms)))

    n_points = 100

    for bi, board in enumerate(args.boards):

        Y, L = [], []

        for fi, framework in enumerate(args.frameworks):

            for ai, alg in enumerate(args.algorithms):

                model_fname = f'{board}_{alg}_auto_{framework}'
                model_fpath = osp.join(ARE_MODEL_DIR, model_fname)
                progress_fpath = osp.join(model_fpath, 'progress.csv')
                print('Progress fpath:', progress_fpath)

                vals = pd.read_csv(progress_fpath, header=None).values
                v = vals[:, col_idx_to_print]

                #y = moving_average(v)
                y = moving_maximum(v)

                best_idx = np.argmax(v)
                print(f'Best results for {alg} on {board}:', vals[best_idx, :])

                best_vals[bi, fi, ai] = v[best_idx]

                n_samples = np.minimum(len(y), n_points)
                if x is None:
                    x = np.arange(n_samples) + 1
                else:
                    assert len(x) == n_samples

                Y.append(y)
                L.append(f'{framework.capitalize()}+{algorithm_names[alg]}')

        y_b = np.ones(len(x)) * baseline_error

        #Y = [y_b] + Y
        #L = ['Baseline (Gaussian)'] + L

        for fsi, fig_size in enumerate(fig_sizes):
            fig, ax = pp.subplots()
            fig.set_size_inches(*fig_size)

            for fi, framework in enumerate(args.frameworks):

                for ai, alg in enumerate(args.algorithms):

                    i = fi * len(args.algorithms) + ai

                    idx = np.arange(0, n_points, int(n_points / n_markers))
                    idx = np.hstack([idx, n_points - 1])

                    pp.plot(x[idx], Y[i][idx], f'{markers[ai]}{line_styles[fi]}', color=colors[ai], linewidth=1.5, label=L[i])
                    #pp.plot(x, Y[i], f'{line_styles[fi]}', color=colors[ai], linewidth=2, label=L[i])

            ram, flash = get_ram_flash(board=board)
            title = f'{board_names[board]} ({ram} KB RAM)'
            #pp.title(title)

            pp.xlim([1, x[-1]])
            pp.xlabel('Iteration')
            pp.ylabel('Accuracy')
            pp.legend()
            #if fsi == 0:
            #    size = "small"
            #elif fsi == 1:
            #    size = "medium"
            #else:
            #    size = "big"
            for ext in ['png', 'pdf']:
                #pp.savefig(osp.join(ARE_FIG_DIR, f'{board}_auto_{",".join(args.frameworks)}_progress_{size}.{ext}'), bbox_inches='tight', pad_inches=0)
                pp.savefig(osp.join(ARE_FIG_DIR, f'{board}_auto_{",".join(args.frameworks)}_progress.{ext}'), bbox_inches='tight', pad_inches=0)
            pp.close()

    for fi, framework in enumerate(args.frameworks):
        for ai, alg in enumerate(args.algorithms):
            row = [f'{framework.upper()}+{alg.upper()}']
            for i, board in enumerate(args.boards):
                if best_vals[i, fi, ai] == np.max(best_vals[i, :]):
                    row.append(f'\\textbf{{{100 * best_vals[i, fi, ai]:.4f}}}')
                else:
                    row.append(f'{100 * best_vals[i, fi, ai]:.4f}')
            print(' & '.join(row))