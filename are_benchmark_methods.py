import json, os
import argparse as arp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.anomaly_detection import *
from utils.preprocess_data import split_data, load_skab, load_odds
from utils.boards import get_ram_flash

from config import RESULT_DIR, BENCHMARK_DATA_DIR, DATA_DIR

def sort_datasets(ds):
    bench = ds['benchmark']
    fname = ds['file']
    fname = fname.split('.csv')[0]
    if bench == 'skab':
        spl = fname.split('_')
        name = spl[0]
        id = ''.join(['0' for _ in range(2 - len(spl[1]))]) + spl[1]
        #print(fname, name, id)
        key = name + id
    else:
        key = fname
    return key

if __name__ == '__main__':

    test_methods = [
        'SKM',
        'CS',
        'WAP',
        'WCM',
        'SOM',
        'GNG',
        'UAE',
        'SAE',
        'DAE',
        'VAE',
        'DSOM',
        'DSVDD',
        #'DSVDD2'
        #'DWCM',
        #'DEM',
        #'DSVDDEM'
    ]

    skip_datasets = [
        'arrhythmia', # odds
        #'cardio',
        #'ionosphere',
        'lympho',
        'mnist',
        'musk',
        'optdigits',
        'speech',
        'wine',

        'other_1',  # skab
        'other_2',
        'other_3',
        'other_4',
        'other_5',
        'other_6',
        'other_7',
        'other_8',
        'other_9',
        'other_10',
        'other_11',
        'other_12',
        'other_13',
        'other_14'
    ]

    parser = arp.ArgumentParser(description='Test AD methods.')
    parser.add_argument('-d', '--dataset', help='Dataset name', default='benchmarks', choices=['benchmarks', 'fan', 'bearing_fft_std'])
    parser.add_argument('-a', '--algorithms', help='Algorithms', default=test_methods, nargs='+', choices=[i for i in HYPERPARAMS.keys()])
    parser.add_argument('-t', '--tries', help='Number of tries', default=3, type=int)
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['flat'])
    parser.add_argument('-n', '--n_samples', help='Number of samples', default=20000, type=int)
    parser.add_argument('-s', '--seed', help='Seed', default=0, type=int)
    parser.add_argument('-g', '--gpu', help='GPU', default='0')
    #parser.add_argument('-r', '--ram', help='Board RAM in Kb', type=float, default=25.6) # 12.8 - nano 40%, 25.6 - nano 80%, 0.8 - uno 40%, 1.6 - uno 80%
    parser.add_argument('-b', '--board', help='Board', default='lora', choices=['uno', 'every', 'iot', 'sense', 'lora'])
    parser.add_argument('-u', '--update_result', help='Update result?', type=bool, default=False)
    parser.add_argument('-l', '--level', help='FPR level', default=0.01, type=float)
    parser.add_argument('-c', '--cut', help='Shorten dataset names?', default=False, type=bool)
    args = parser.parse_args()

    if args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ram, flash = get_ram_flash(args.board)

    precision = 4

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    dataset = args.dataset
    n_tries = args.tries

    if not osp.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    result_fname = f'{args.board}_accuracy_{dataset}.json'
    result_fpath = osp.join(RESULT_DIR, result_fname)
    try:
        with open(result_fpath, 'r') as f:
            result_data = json.load(f)
    except Exception as e:
        print(e)
        result_data = {}

    target_datasets = {}

    if dataset == 'benchmarks':

        dataset_files = os.listdir(BENCHMARK_DATA_DIR)
        with open(f'{BENCHMARK_DATA_DIR}/metainfo.json', 'r') as fp:
            metainfo = json.load(fp)

        bench_stats = {}

        benchmarks = [dataset['benchmark'] for dataset in metainfo]
        for bench in sorted(list(set(benchmarks))): #[::-1]:

            bench_stats[bench] = {
                'n_datasets': 0,
                'n_samples_min': np.inf,
                'n_samples_max': 0,
                'n_features_min': np.inf,
                'n_features_max': 0
            }

            for dataset in sorted(metainfo, key=sort_datasets):
                if dataset['benchmark'] == bench:
                    fpath = osp.join(BENCHMARK_DATA_DIR, dataset['file'])
                    loader = globals()[f"load_{bench}"]
                    ds = loader(fpath, n_samples=args.n_samples)
                    print(dataset, ds['X'].shape)
                    ds_name = dataset['file'].split('.')[0]
                    target_datasets[ds_name] = ds

                    if ds_name not in skip_datasets:
                        bench_stats[bench]['n_datasets'] += 1
                        if ds['X'].shape[0] < bench_stats[bench]['n_samples_min']:
                            bench_stats[bench]['n_samples_min'] = ds['X'].shape[0]
                        if ds['X'].shape[0] > bench_stats[bench]['n_samples_max']:
                            bench_stats[bench]['n_samples_max'] = ds['X'].shape[0]
                        if ds['X'].shape[1] < bench_stats[bench]['n_features_min']:
                            bench_stats[bench]['n_features_min'] = ds['X'].shape[1]
                        if ds['X'].shape[1] > bench_stats[bench]['n_features_max']:
                            bench_stats[bench]['n_features_max'] = ds['X'].shape[1]

        print(bench_stats)

    else:

        #data_fpath = osp.join(DATA_DIR, dataset)
        #target_dataset = load_dataset(data_fpath, series_len=1, series_step=1, labels=labels, feature_extractors=args.feature_extractors, n_samples=args.n_samples)
        #target_datasets[dataset] = target_dataset

        pass

    for di, dataset_name in enumerate(target_datasets.keys()):

        print(di + 1, '/', len(target_datasets.keys()), ':', dataset_name)

        #if di < 24:
        #    print(f'Skipping {dataset_name}...')
        #    continue

        if dataset_name in skip_datasets:
            print(f'Skipping {dataset_name}...')
            continue

        if dataset_name not in result_data.keys():
            result_data[dataset_name] = {}

        target_dataset = target_datasets[dataset_name]

        data = split_data(target_dataset, shuffle_features=False, inf_split=0.2)

        print('Training data shape:', data['tr'][0].shape, data['tr'][1].shape)
        print('Validation data shape:', data['val'][0].shape, data['val'][1].shape)
        print('Inference data shape:', data['inf'][0].shape, data['inf'][1].shape)

        acc_best, tpr_best, fpr_best, metric_best, method_best, n_clusters_best = 0, 0, 0, 0, None, 0

        for method in args.algorithms:

            print(method)

            if method not in result_data[dataset_name].keys() or args.update_result or result_data[dataset_name][method] < 0:
            #elif method not in result_data[dataset_name].keys() or args.update_result:

                method_class = locals()[method]
                m = method_class()
                acc_method = -1.0

                if len(HYPERPARAMS[method]) == 2 and type(HYPERPARAMS[method][0]) is list and type(HYPERPARAMS[method][1]) is list:
                    hps = []
                    for hp0 in HYPERPARAMS[method][0]:
                        for hp1 in HYPERPARAMS[method][1]:
                            hps.append([hp0, hp1])
                else:
                    hps = HYPERPARAMS[method]

                for hp in hps:

                    try:
                    #if 1:

                        acc_sum, fpr_sum, tpr_sum, auc_sum, metric_sum = 0, 0, 0, 0, 0
                        for j in range(n_tries):

                            np.random.seed(args.seed + j + 1)
                            tf.random.set_seed(args.seed + j + 1)

                            alpha, metric_val = m.fit(data['tr'], validation_data=data['val'], hp=hp, mem_max=ram)

                            acc, fpr, tpr, auc, preds = m.evaluate(data['inf'], alpha, args.level)

                            data1_01 = data['inf'][1].copy()
                            data1_01[np.where(data1_01 > 0)] = 1
                            fpr, tpr, thresholds = roc_curve(data1_01, preds)
                            thresholds_lvl = [thr for thr, fpr_val in zip(thresholds, fpr) if fpr_val <= args.level]
                            thresholds_uniq = np.unique(thresholds_lvl)
                            acc_per_thr = np.zeros(len(thresholds_uniq))
                            for i, threshold in enumerate(thresholds_uniq):
                                preds_01 = np.zeros(len(preds))
                                preds_01[np.where(preds > threshold)] = 1
                                acc_per_thr[i] = float(len(np.where(preds_01 == data1_01)[0])) / len(preds)

                            acc = np.max(acc_per_thr)
                            # print(acc)
                            acc_sum += acc

                        if acc_sum / n_tries > acc_method:
                            acc_method = acc_sum / n_tries

                    except Exception as e:
                    #else:
                        print(hp, e)

                if method in result_data[dataset_name].keys():
                    print('\n')
                    print('before:', result_data[dataset_name][method])
                    print('after:', acc_method)
                    print('\n')

                result_data[dataset_name][method] = acc_method

                with open(result_fpath, 'w') as f:
                    json.dump(result_data, f)

        print('Done:', result_data[dataset_name])

    hat = ['Dataset'] + args.algorithms
    hat = ' & '.join(hat) + ' \\\\ \hline'

    print(hat)

    ranks = {
        'total': {}
    }

    accs = {
        'total': {}
    }

    n_datasets = {'total': 0}

    for benchmark in list(set(benchmarks)):
        ranks[benchmark] = {}
        accs[benchmark] = {}
        n_datasets[benchmark] = 0

    for method in args.algorithms:
        for key in ranks.keys():
            ranks[key][method] = 0
            accs[key][method] = 0

    for di, dataset_name in enumerate(target_datasets.keys()):

        benchmark = [item['benchmark'] for item in metainfo if item['file'].split('.')[0] == dataset_name][0]

        if dataset_name in result_data.keys():

            result_row = result_data[dataset_name]
            result_array = np.array([val for key, val in result_row.items() if key in args.algorithms])

            if dataset_name in skip_datasets:
                # print(f'Skipping {dataset_name}...')
                # print([key for key, value in result_row.items() if value < 0])
                # print(target_datasets[dataset_name]['X'].shape)
                continue

            if np.any(result_array < 0):
                print('\n***************\n')
                print(dataset_name)
                print(result_array)
                print('\n***************\n')

            n_datasets[benchmark] += 1
            n_datasets['total'] += 1

            if args.cut:
                if dataset_name.startswith('valve'):
                    postfix = dataset_name.split('valve')[1]
                    postfix = postfix.replace('_', '\_')
                    ds_name = f'v{postfix}'
                elif dataset_name.startswith('other'):
                    postfix = dataset_name.split('other')[1]
                    postfix = postfix.replace('_', '\_')
                    ds_name = f'ot{postfix}'
                elif dataset_name.startswith('mammo'):
                    ds_name = f'{dataset_name[:3]}.'
                elif len(dataset_name) <= 5:
                    ds_name = dataset_name
                else:
                    ds_name = f'{dataset_name[:4]}.'
            else:
                if dataset_name.startswith('valve'):
                    postfix = dataset_name.split('valve')[1]
                    postfix = postfix.replace('_', '\_')
                    ds_name = f'valve{postfix}'
                elif dataset_name.startswith('other'):
                    postfix = dataset_name.split('other')[1]
                    postfix = postfix.replace('_', '\_')
                    ds_name = f'other{postfix}'
                else:
                    ds_name = dataset_name

            row = [ds_name]
            acc_best = np.max([val for val in result_row.values()])
            acc_worst = np.min([val for val in result_row.values()])
            for method in args.algorithms:
                if method in result_row.keys():
                    acc_val = result_row[method] * 100
                    if acc_val == acc_best * 100:
                        row.append(f'\\textbf{{{acc_val:.{precision}f}}}')
                    else:
                        row.append(f'{acc_val:.{precision}f}')
                    ranks[benchmark][method] += (result_row[method] - acc_worst) / (acc_best - acc_worst + 1e-10)
                    ranks['total'][method] += (result_row[method] - acc_worst) / (acc_best - acc_worst + 1e-10)
                    accs[benchmark][method] += result_row[method]
                    accs['total'][method] += result_row[method]
            print(' & '.join(row) + ' \\\\')

    print()

    for benchmark in list(set(benchmarks)) + ['total']:
        rank_best = np.max([item for item in ranks[benchmark].values()])
        acc_best = np.max([item for item in accs[benchmark].values()])
        #print(f'{benchmark.upper()} ranks:')
        rank_row = [benchmark.upper()]
        acc_row = [benchmark.upper()]
        for method in args.algorithms:
            #print(f'{method}: {ranks[benchmark][method] / n_datasets[benchmark]:.4f}')
            if ranks[benchmark][method] == rank_best:
                rank_row.append(f'\\textbf{{{ranks[benchmark][method] / n_datasets[benchmark]:.{precision}f}}}')
            else:
                rank_row.append(f'{ranks[benchmark][method] / n_datasets[benchmark]:.{precision}f}')
            if accs[benchmark][method] == acc_best:
                acc_row.append(f'\\textbf{{{accs[benchmark][method] / n_datasets[benchmark] * 100:.{precision}f}}}')
            else:
                acc_row.append(f'{accs[benchmark][method] / n_datasets[benchmark] * 100:.{precision}f}')
        #print(' & '.join(rank_row) + ' \\\\')
        print(' & '.join(acc_row) + ' \\\\')