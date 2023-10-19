import os

import numpy as np
import argparse as arp
import tensorflow as tf

from are_plot_rocs import CS, SKM, SOM, DSVDD
from utils.preprocess_data import load_dataset, split_data
from utils.boards import get_ram_flash
from config import *
from matplotlib import pyplot as pp
from matplotlib.colors import BASE_COLORS, CSS4_COLORS
from matplotlib.markers import MarkerStyle as ms

def plot_models():
    dataset = args.dataset
    if dataset.startswith('fan'):
        labels = {0: ['normal', 'on_off'], 1: ['stick'], 2: ['tape'], 3: ['shake']}
    elif dataset.startswith('bearing'):
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(ARE_DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=1, series_step=1, labels=labels, feature_extractors=args.feature_extractors, n_samples=args.n_samples)
    data = split_data(target_dataset, shuffle_features=False, inf_split=0.2)

    print('Training data shape:', data['tr'][0].shape)
    print('Validation data shape:', data['val'][0].shape)
    print('Inference data shape:', data['inf'][0].shape)

    hps = {
        'skm': [6, 4],
        'som': [36, 36],
        'dsvdd': [2, 64]
    }

    titles = {
        'skm': 'Scalable k-means++',
        'som': 'Self-organising map',
        'dsvdd': 'Deep support vector data description'
    }

    ram, flash = get_ram_flash(board=args.board)

    method_class = locals()[args.algorithm.upper()]
    m = method_class()
    alpha, metric_val = m.fit(data['tr'], validation_data=data['val'], hp=hps[args.algorithm], mem_max=ram)

    prefix = args.algorithm
    title = titles[args.algorithm]

    m.plot(data['inf'], fig_dir=ARE_FIG_DIR, prefix=prefix, title=title)

def plot_speed(board, algorithms):

    pp.style.use('default')
    colormap = 'Dark2'

    fig_size = (7, 4.3)
    n_markers = 9

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

    algorithm_names = [
        'SKM',  # 'Scalable k-means++',
        'CS',  # 'CluStrem',
        'DSVDD'  # 'Deep support vector data description'
    ]

    dpath = osp.join(SKETCH_TEST_DIR, board)
    N_tr = []
    N_inf = []

    for algorithm in algorithms:
        fpath = None
        for fname in os.listdir(dpath):
            if fname.startswith(algorithm):
                fpath = osp.join(dpath, fname)
                break
        if fpath is not None:
            print(fpath)
            with open(fpath, 'r') as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            line_tr_idx = lines.index('Training:')
            line_inf_idx = lines.index('Inferencing:')
            line_done_idx = lines.index('Done!')
            tr_lines = [lines[i].split(',') for i in range(line_tr_idx + 1, line_inf_idx)]
            inf_lines = [lines[i].split(',') for i in range(line_inf_idx + 1, line_done_idx)]
            n_tr = np.array(tr_lines, dtype=float)
            n_tr[:, 0] = n_tr[:, 0] - n_tr[0, 0]
            n_inf = np.array(inf_lines, dtype=float)
            n_inf[:, 0] = n_inf[:, 0] - n_inf[0, 0]

            idx_tr = np.where(n_tr[:, 0] <= 60 * 1000)[0]
            idx_inf = np.where(n_inf[:, 0] <= 60 * 1000)[0]

            N_tr.append(n_tr[idx_tr, :])
            N_inf.append(n_inf[idx_inf, :])

    results = {}

    for stage, array in zip(['training', 'inference'], [N_tr, N_inf]):

        fig, ax = pp.subplots()
        fig.set_size_inches(*fig_size)

        results[stage] = {}

        for i, n in enumerate(array):

            results[stage][algorithms[i]] = n[-1, 1] / n[-1, 0] * 1000

            n_points = n.shape[0]
            idx = np.arange(0, n_points, int(n_points / n_markers))
            idx = np.hstack([idx, n_points - 1])
            ax.plot(n[idx, 0] / 1000, n[idx, 1], f'{markers[i]}-', color=colors[i], linewidth=1.5, label=algorithm_names[i])
        pp.legend()
        pp.xlim([0, 60])
        pp.xlabel('Time, seconds')
        pp.ylabel(f'Number of the samples processed ({stage})')
        pp.legend()
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(ARE_FIG_DIR, f'{board}_{stage}_rate.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

    return results

def plot_power(board, algorithms, fname='power_measurements.txt'):

    pp.style.use('default')
    colormap = 'Dark2'

    fig_size = (7, 4.3)
    n_markers = 9

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

    algorithm_names = [
        'SKM',  # 'Scalable k-means++',
        'CS',  # 'CluStrem',
        'DSVDD'  # 'Deep support vector data description'
    ]

    fpath = osp.join(SKETCH_TEST_DIR, fname)

    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    if board in lines:
        brd_idx = lines.index(board)
        lines = lines[brd_idx + 1 :]

        dpath = osp.join(SKETCH_TEST_DIR, board)

        power_limits = {}
        ns = {}

        for algorithm in algorithms:
            if algorithm in lines:
                line_alg_idx = lines.index(algorithm)
                power_limits[algorithm] = np.array([float(lines[i + 1]) for i in range(line_alg_idx, line_alg_idx + 4)])

            fpath = None
            for fname in os.listdir(dpath):
                if fname.startswith(algorithm):
                    fpath = osp.join(dpath, fname)
                    break
            if fpath is not None:
                print(fpath)
                with open(fpath, 'r') as f:
                    rate_lines = f.readlines()
                rate_lines = [line.strip() for line in rate_lines]
                line_tr_idx = rate_lines.index('Training:')
                line_inf_idx = rate_lines.index('Inferencing:')
                line_done_idx = rate_lines.index('Done!')
                tr_lines = [rate_lines[i].split(',') for i in range(line_tr_idx + 1, line_inf_idx)]
                inf_lines = [rate_lines[i].split(',') for i in range(line_inf_idx + 1, line_done_idx)]
                n_tr = np.array(tr_lines, dtype=float)
                n_tr[:, 0] = n_tr[:, 0] - n_tr[0, 0]
                n_inf = np.array(inf_lines, dtype=float)
                n_inf[:, 0] = n_inf[:, 0] - n_inf[0, 0]

                idx_tr = np.where(n_tr[:, 0] <= 60 * 1000)[0]
                idx_inf = np.where(n_inf[:, 0] <= 60 * 1000)[0]

                ns[algorithm] = [n_tr[idx_tr[-1], 1], n_inf[idx_inf[-1], 1]]

    print(power_limits)
    print(ns)

    results = {}

    n_samples = 60

    print(power_limits.keys())

    for stage_i, stage in enumerate(['training', 'inference']):

        fig, ax = pp.subplots()
        fig.set_size_inches(*fig_size)

        results[stage] = {}

        for i, alg in enumerate(algorithms):

            limit_vals = power_limits[alg][stage_i * 2 : (stage_i + 1) * 2]
            p_min = np.min(limit_vals)
            p_max = np.max(limit_vals)

            x = np.arange(n_samples + 1)

            #vals = np.cumsum(np.hstack([0, np.random.uniform(p_min, p_max, n_samples) * 5 / 1000]))
            #vals = np.random.uniform(p_min, p_max, n_samples + 1) * 5 / 1000
            vals = np.interp(x, [x[0], x[-1]], [limit_vals[0] * 5 / 1000, limit_vals[-1] * 5 / 1000])

            #print(board, alg, vals)

            #results[stage][alg] = vals[-1] / ns[alg][stage_i]
            results[stage][alg] = np.sum(vals) / ns[alg][stage_i]

            n_points = vals.shape[0]
            idx = np.arange(0, n_points, int(n_points / n_markers))
            idx = np.hstack([idx, n_points - 1])
            ax.plot(x[idx], vals[idx], f'{markers[i]}-', color=colors[i], linewidth=1.5, label=algorithm_names[i])

        pp.legend()
        pp.xlim([0, 60])
        pp.xlabel('Time, seconds')
        pp.ylabel(f'Power, watts ({stage})')
        pp.legend()
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(ARE_FIG_DIR, f'{board}_{stage}_power.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()

    return results

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Plot AD models.')
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing_fft_std_rnd')
    parser.add_argument('-a', '--algorithms', help='Algorithms', nargs='+', default=['skm', 'cs', 'dsvdd'])
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['flat'])
    parser.add_argument('-n', '--n_samples', help='Number of samples', default=5000, type=int)
    parser.add_argument('-s', '--seed', help='Seed', default=0, type=int)
    parser.add_argument('-b', '--boards', help='Board names', nargs='+', default=['uno', 'every', 'iot', 'sense', 'lora'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    #plot_models()

    rate_results = {}
    power_results = {}

    for board in args.boards:
        #rate_results[board] = plot_speed(board, args.algorithms)
        power_results[board] = plot_power(board, args.algorithms)

    results = power_results
    print(results)

    board_key = args.boards[0]
    alg_key = args.algorithms[0]
    stage_key = [key for key in results[board_key].keys()][0]

    for stage in results[board_key].keys():
        values = []
        for alg in results[board_key][stage_key].keys():
            values.append([])
            for board in results.keys():
                values[-1].append(results[board][stage][alg])
        values = np.array(values)
        #best_values = np.max(values, 0)
        best_values = np.min(values, 0)
        #print(best_values)
        for i, a in enumerate(args.algorithms):
            row = [stage.capitalize(), a.upper()]
            for j, v in enumerate(values[i, :]):
                if v == best_values[j]:
                    row.append(f'\\textbf{{{v * 1000:.4f}}}')
                else:
                    row.append(f'{v * 1000:.4f}')
            print(' & '.join(row) + ' \\\\')
        print('\midrule')





