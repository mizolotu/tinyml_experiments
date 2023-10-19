import argparse as arp

try:
    from matplotlib import pyplot as pp
    from matplotlib.colors import BASE_COLORS, CSS4_COLORS, Normalize
    from matplotlib.markers import MarkerStyle as ms
    from matplotlib.ticker import FormatStrFormatter
except Exception as e:
    print(e)
    print('Cannot import matplotlib :(')

from utils.preprocess_data import load_dataset, split_data
from config import *

from utils.anomaly_detection import *
from utils.boards import get_ram_flash


if __name__ == '__main__':

    test_methods = [

        'SKM',
        'CS',
        'WAP',
        'WCM',
        'SOM',

        #'GNG',
        #'VAE',
        #'UAE',
        #'SAE',
        #'DAE',
        #'DSOM',

        'DSVDD',

        #'DWCM',
        #'DEM'
    ]

    parser = arp.ArgumentParser(description='Test AD methods.')
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing_fft_std_rnd', choices=['fan', 'bearing_fft_std_rnd'])
    parser.add_argument('-a', '--algorithms', help='Algorithms', default=test_methods, nargs='+', choices=[i for i in HYPERPARAMS.keys()])
    parser.add_argument('-t', '--tries', help='Number of tries', default=3, type=int)
    parser.add_argument('-m', '--metric', help='Metric', default='em', choices=['em', 'mv'])
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['flat'])
    parser.add_argument('-n', '--n_samples', help='Number of samples', default=None, type=int)
    parser.add_argument('-s', '--seed', help='Seed', default=0, type=int)
    parser.add_argument('-p', '--plot', help='Plot?', type=bool, default=False)
    parser.add_argument('-g', '--gpu', help='GPU', default='-1')
    #parser.add_argument('-r', '--ram', help='Board RAM in Kb', type=float, nargs='+', default=[51.2]) # 25.6,19.2,12.8,1.6,1.2,0.8
    parser.add_argument('-b', '--board', help='Board', default='uno', choices=['uno', 'every', 'iot', 'sense', 'lora'])
    parser.add_argument('-l', '--level', help='FPR level', default=0.05, type=float)
    args = parser.parse_args()

    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    dataset = args.dataset
    if dataset.startswith('fan'):
        #labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
        labels = {0: ['normal', 'on_off'], 1: ['stick'], 2: ['tape'], 3: ['shake']}
    elif dataset.startswith('bearing'):
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(ARE_DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=1, series_step=1, labels=labels, feature_extractors=args.feature_extractors, n_samples=args.n_samples)
    data = split_data(target_dataset, shuffle_features=False, inf_split=0.2)

    print('Training data shape:', data['tr'][0].shape)
    print('Validation data shape:', data['val'][0].shape)
    print('Inference data shape:', data['inf'][0].shape)

    n_tries = args.tries

    fpr_lims = [0.05]
    tpr_lims = [0 for _ in fpr_lims]

    ram, flash = get_ram_flash(args.board)

    acc_best, tpr_best, fpr_best, metric_best, method_best, n_clusters_best = 0, 0, 0, 0, None, 0

    roc_data = []
    for method in args.algorithms:

        print(f'Evaluating {method}...')

        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)

        method_class = locals()[method]
        m = method_class()
        acc_method = 0
        auc_method = 0
        preds_method = []

        if len(HYPERPARAMS[method]) == 2 and type(HYPERPARAMS[method][0]) is list and type(HYPERPARAMS[method][1]) is list:
            hps = []
            for hp0 in HYPERPARAMS[method][0]:
                for hp1 in HYPERPARAMS[method][1]:
                    hps.append([hp0, hp1])
        else:
            hps = HYPERPARAMS[method]

        for hp in hps:
            acc_sum, fpr_sum, tpr_sum, auc_sum, metric_sum = 0, 0, 0, 0, 0
            preds_per_try = []
            for j in range(n_tries):

                np.random.seed(args.seed + j)
                tf.random.set_seed(args.seed + j)
                tf.keras.utils.set_random_seed(args.seed + j)

                try:
                #if 1:

                    alpha, metric_val = m.fit(data['tr'], validation_data=data['val'], hp=hp, mem_max=ram, metric=args.metric, adjust_params=False)

                    acc, fpr, tpr, auc, preds = m.evaluate(data['inf'], alpha, args.level)
                    acc_sum += acc
                    fpr_sum += fpr
                    tpr_sum += tpr
                    auc_sum += auc
                    metric_sum += metric_val
                    preds_per_try.append(preds)

                except Exception as e:
                #else:
                    #pass
                    print(e)

            if acc_sum / n_tries > acc_method:
            #if auc_sum / n_tries > auc_method:
                acc_method = acc_sum / n_tries
                # auc_method = auc_sum / n_tries
                preds_method = np.hstack(preds_per_try)

            if method_best is None:
                update_best = True
            elif args.metric == 'em' and metric_sum / n_tries > metric_best:
                update_best = True
            elif args.metric == 'mv' and metric_sum / n_tries < metric_best:
                update_best = True
            else:
                update_best = False

            if update_best:
                metric_best = metric_sum / n_tries
                acc_best = acc_sum / n_tries
                tpr_best = tpr_sum / n_tries
                fpr_best = fpr_sum / n_tries
                method_best = m.__class__.__name__
                hp_best = hp

            if args.plot:
                m.tsne_plot(data['inf'], prefix=f'{dataset}_{m.__class__.__name__}_{hp}', labels=[label_list[0] for label_list in labels.values()], n_samples=10000)

            print(f'{m.__class__.__name__} with hyperparameter {hp} on average: acc = {acc_sum / n_tries}, fpr = {fpr_sum / n_tries}, tpr = {tpr_sum / n_tries}, {args.metric} = {metric_sum / n_tries}')

        data1_01 = data['inf'][1].copy()
        data1_01[np.where(data1_01 > 0)] = 1

        fpr, tpr, thresholds = roc_curve(np.tile(data1_01, n_tries), preds_method)
        roc_data.append([m.__class__.__name__, fpr, tpr])
        for fpr_lim_i, fpr_lim in enumerate(fpr_lims):
            idx = np.where(fpr < fpr_lim)[0]
            if len(idx) > 0:
                #potential_tpr_lim = tpr[idx[-1] + 1] if idx[-1] < len(tpr) - 1 else tpr[idx[-1]]
                potential_tpr_lim = tpr[idx[-1]]
                if potential_tpr_lim > tpr_lims[fpr_lim_i]:
                    tpr_lims[fpr_lim_i] = potential_tpr_lim

    print(f'The best is {method_best} with hyperparameter {hp_best}: acc = {acc_best}, fpr = {fpr_best}, tpr = {tpr_best}, {args.metric} = {metric_best}')

    pp.style.use('default')
    colormap = 'Dark2'

    #colors = ['orangered', 'teal', 'darkmagenta', 'midnightblue']
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

    n_markers = 40

    fig_size = (7, 4.3) # (9, 5.8)

    for xlim, ylim in zip(fpr_lims, tpr_lims):

        fig, ax = pp.subplots()
        fig.set_size_inches(*fig_size)

        for i, (name, fpr, tpr) in enumerate(roc_data):
            #pp.plot(fpr, tpr, color=colors[i], linewidth=1.5, label='Clustering' if name == 'CluStream' else name)
            idx_last = np.where(fpr > xlim)[0]
            if len(idx_last) == 0:
                idx_last = -1
            else:
                idx_last = idx_last[0]
            fpr = fpr[:idx_last]
            tpr = tpr[:idx_last]
            n = len(tpr)
            idx = np.arange(0, n, int(n / n_markers))
            idx = np.hstack([idx, n - 1])
            pp.plot(fpr[idx], tpr[idx], f'{markers[i]}-', color=colors[i], linewidth=1.5, label=name)
            #pp.plot(fpr, tpr, f'{markers[i]}-', color=colors[i], linewidth=1.5, label=name)

        pp.xlim([0, xlim])
        pp.ylim([0, ylim])
        pp.xlabel('False positive rate')
        pp.ylabel('True positive rate')
        pp.legend()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for ext in ['png', 'pdf']:
            pp.savefig(osp.join(ARE_FIG_DIR, f'{args.board}_roc_{dataset}_{args.level}.{ext}'), bbox_inches='tight', pad_inches=0)
        pp.close()