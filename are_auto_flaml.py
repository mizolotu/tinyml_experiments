import numpy as np
import json, joblib
import argparse as arp
import tensorflow as tf

from config import *
from flaml import BlendSearch, tune
from utils.preprocess_data import load_dataset, split_data
from utils.boards import get_ram_flash
from are_plot_rocs import CS, SKM, SOM, DSVDD
from sklearn.metrics import roc_auc_score, roc_curve

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Train auto (blend search) model to detect anomalous vibrations.')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=42)
    parser.add_argument('-n', '--ntries', help='Number of tries', type=int, default=3)
    parser.add_argument('-r', '--sampling_rate', help='Sampling rate', type=float, default=1.0)
    parser.add_argument('-v', '--verbose', help='Verbosity', type=int, default=0)
    parser.add_argument('-t', '--trials', help='Number of trials', type=int, default=100)
    parser.add_argument('-b', '--boards', help='Board names', nargs='+', default=['uno', 'every', 'iot', 'sense', 'lora'])  # ['uno', 'every', 'iot', 'sense', 'lora']
    parser.add_argument('-d', '--dataset', help='Dataset name', default='bearing_fft_std_rnd', choices=['fan', 'bearing_fft_std_rnd'])
    parser.add_argument('-a', '--algorithm', help='Algorithm', default='dsvdd', choices=['cs', 'skm', 'som', 'dsvdd'])
    parser.add_argument('-l', '--level', help='FPR level', default=0.05, type=float)
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['flat'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    seed = args.seed

    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    dataset = args.dataset

    ntries = args.ntries

    preproc_size = 0.6

    val_split = 0.2
    output_dim = 1
    loss = 'mean_squared_error'
    epochs = 1000
    patience = 10
    lr = 1e-3
    batch_size = 4096
    dropout = 0.5
    metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
    alpha = 3

    dataset = args.dataset
    if dataset.startswith('fan'):
        # labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
        labels = {0: ['normal', 'on_off'], 1: ['stick'], 2: ['tape'], 3: ['shake']}
    elif dataset.startswith('bearing'):
        labels = {0: ['normal'], 1: ['crack', 'sand']}

    data_fpath = osp.join(ARE_DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=1, series_step=1, labels=labels, feature_extractors=args.feature_extractors)
    data = split_data(target_dataset, shuffle_features=False, inf_split=0.2)

    print('Training data shape:', data['tr'][0].shape)
    print('Validation data shape:', data['val'][0].shape)
    print('Inference data shape:', data['inf'][0].shape)

    x_mean = np.mean(data['tr'][0], 0)
    x_std = np.std(data['tr'][0], 0)
    x_min = np.min(data['tr'][0], 0)
    x_max = np.max(data['tr'][0], 0)
    y_min = np.min(data['tr'][0], 0)
    y_max = np.max(data['tr'][0], 0)

    input_shape = data['tr'][0].shape[1:]
    output_dim = 1

    # baseline model

    X_tr_std = (data['tr'][0] - x_min[None, :]) / (x_max[None, :] - x_min[None, :] + 1e-10)
    X_val_std = (data['val'][0] - x_min[None, :]) / (x_max[None, :] - x_min[None, :] + 1e-10)
    X_inf_std = (data['inf'][0] - x_min[None, :]) / (x_max[None, :] - x_min[None, :] + 1e-10)

    centroid = np.mean(X_tr_std, 0)

    D_val = np.sum((centroid[None, :] - X_val_std) ** 2, axis=-1)
    d_max = np.max(D_val)
    d_std = np.std(D_val)
    thr = d_max + alpha * d_std
    D_inf = np.sum((centroid[None, :] - X_inf_std) ** 2, axis=-1).flatten()
    predictions = np.zeros(len(data['inf'][1]))
    predictions[np.where(D_inf > thr)[0]] = 1

    data1_01 = data['inf'][1].copy()
    data1_01[np.where(data1_01 > 0)] = 1

    probs = D_inf / thr

    fpr, tpr, thresholds = roc_curve(data1_01, probs)
    thresholds_lvl = [thr for thr, fpr_val in zip(thresholds, fpr) if fpr_val <= args.level]
    thresholds_uniq = np.unique(thresholds_lvl)
    acc_per_thr = np.zeros(len(thresholds_uniq))
    for i, threshold in enumerate(thresholds_uniq):
        preds_01 = np.zeros(len(probs))
        preds_01[np.where(probs > threshold)] = 1
        acc_per_thr[i] = float(len(np.where(preds_01 == data1_01)[0])) / len(probs)

    acc_baseline = np.max(acc_per_thr)

    fpr_baseline = len(np.where((predictions == 1) & (data1_01 == 0))[0]) / (1e-10 + len(np.where(data1_01 == 0)[0]))
    tpr_baseline = len(np.where((predictions == 1) & (data1_01 == 1))[0]) / (1e-10 + len(np.where(data1_01 == 1)[0]))
    auc_baseline = roc_auc_score(data1_01, predictions)

    print('Acc baseline:', acc_baseline, 'FPR baseline:', fpr_baseline, 'TPR baseline:', tpr_baseline, 'AUC baseline:', auc_baseline)

    baseline_model_fname = 'baseline_cluster1'
    baseline_model_fpath = osp.join(ARE_MODEL_DIR, baseline_model_fname)
    print('Baseline model fpath:', baseline_model_fpath)

    if not osp.isdir(baseline_model_fpath):
        os.mkdir(baseline_model_fpath)

    with open(osp.join(baseline_model_fpath, 'metainfo.json'), 'w') as f:
        json.dump({
            'input_shape': input_shape,
            'output_shape': output_dim,
            'x_mean': x_mean.tolist(),
            'x_std': x_std.tolist(),
            'x_min': x_min.tolist(),
            'x_max': x_max.tolist()
        }, f)

    with open(osp.join(baseline_model_fpath, 'stats.json'), 'w') as f:
        json.dump({
            'acc': acc_baseline,
            'fpr': fpr_baseline,
            'tpr': tpr_baseline,
            'auc': auc_baseline
        }, f)

    for board in args.boards:

        #model_fname = f'{board}_{args.algorithm}_auto_{mode if mode is not None else "all-samples"}_{args.label}'
        model_fname = f'{board}_{args.algorithm}_auto_flaml'
        model_fpath = osp.join(ARE_MODEL_DIR, model_fname)
        print('Model fpath:', model_fpath)

        if not osp.isdir(model_fpath):
            os.mkdir(model_fpath)

        with open(osp.join(model_fpath, 'metainfo.json'), 'w') as f:
            json.dump({
                'input_shape': input_shape,
                'output_shape': output_dim,
                'x_min': x_min.tolist(),
                'x_max': x_max.tolist(),
            }, f)

        # main part

        progress = []
        ram, flash = get_ram_flash(board=board)

        def train_cs_model(config):

            # sampled data

            n_clusters = int(np.round(config['n_clusters']))
            p_microclusters = config['p_microclusters']
            n_microclusters = int(np.round(n_clusters * p_microclusters))

            model_size = preproc_size + (2 * np.prod(input_shape) * n_clusters + np.prod(input_shape) * n_microclusters) * 4 / 1024
            print('The model size to the ram ratio:', model_size / ram)

            if model_size > ram:
                #acc, fpr, tpr, auc = acc_baseline, fpr_baseline, tpr_baseline, auc_baseline
                acc, fpr, tpr, auc = 0.5, 0.0, 0.0, 0.5
                print('*********************')
                print('The model is too big!')
                print('*********************')

            else:

                try:

                    acc, fpr, tpr, auc, preds = 0, 0, 0, 0, []

                    for j in range(ntries):

                        np.random.seed(seed + j)
                        tf.random.set_seed(seed + j)
                        tf.keras.utils.set_random_seed(seed + j)

                        model = CS()
                        alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_clusters, p_microclusters], mem_max=ram, adjust_params=False)
                        acc_, fpr_, tpr_, auc_, preds_ = model.evaluate(data['inf'], alpha, args.level)

                        acc += acc_
                        fpr += fpr_
                        tpr += tpr_
                        auc += auc_
                        preds.extend(preds_)

                    acc /= ntries
                    fpr /= ntries
                    tpr /= ntries
                    auc /= ntries

                except Exception as e:
                    print(e)
                    acc, fpr, tpr, auc = 0.5, 0.0, 0.0, 0.5

                print('params:', config, 'acc:', acc)

            progress.append(f'{acc},{fpr},{tpr},{auc},{config["n_clusters"]},{config["p_microclusters"]}\n')

            return {"acc": acc, 'fpr': fpr, 'tpr': tpr, 'auc': auc}

        def train_skm_model(config):

            # sampled data

            n_clusters = int(np.round(config['n_clusters']))
            p_batch_size = config['p_batch_size']
            n_batch_size = int(np.round(n_clusters * p_batch_size))

            model_size = preproc_size + (2 * np.prod(input_shape) * n_clusters + np.prod(input_shape) * n_batch_size) * 4 / 1024
            print('The model size to the ram ratio:', model_size / ram)

            if model_size > ram:
                #acc, fpr, tpr, auc = acc_baseline, fpr_baseline, tpr_baseline, auc_baseline
                acc, fpr, tpr, auc = 0.5, 0.0, 0.0, 0.5
                print('*********************')
                print('The model is too big!')
                print('*********************')

            else:

                try:

                    acc, fpr, tpr, auc, preds = 0, 0, 0, 0, []

                    for j in range(ntries):

                        np.random.seed(seed + j)
                        tf.random.set_seed(seed + j)
                        tf.keras.utils.set_random_seed(seed + j)

                        model = SKM()
                        alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_clusters, p_batch_size], mem_max=ram, adjust_params=False)
                        acc_, fpr_, tpr_, auc_, preds_ = model.evaluate(data['inf'], alpha, args.level)

                        acc += acc_
                        fpr += fpr_
                        tpr += tpr_
                        auc += auc_
                        preds.extend(preds_)

                    acc /= ntries
                    fpr /= ntries
                    tpr /= ntries
                    auc /= ntries

                except Exception as e:
                    print(e)
                    acc, fpr, tpr, auc = 0.5, 0.0, 0.0, 0.5

                print('params:', config, 'acc:', acc)

            progress.append(f'{acc},{fpr},{tpr},{auc},{config["n_clusters"]},{config["p_batch_size"]}\n')

            return {"acc": acc, 'fpr': fpr, 'tpr': tpr, 'auc': auc}

        def train_som_model(config):

            # sampled data

            map_height = int(np.round(config['map_height']))
            map_width = int(np.round(config['map_width']))

            model_size = preproc_size + np.prod(input_shape) * map_width * map_height * 4 / 1024
            print('The model size to the ram ratio:', model_size / ram)

            if model_size > ram:
                #acc, fpr, tpr, auc = acc_baseline, fpr_baseline, tpr_baseline, auc_baseline
                acc, fpr, tpr, auc = 0.5, 0.0, 0.0, 0.5
                print('*********************')
                print('The model is too big!')
                print('*********************')

            else:

                try:

                    acc, fpr, tpr, auc, preds = 0, 0, 0, 0, []

                    for j in range(ntries):

                        np.random.seed(seed + j)
                        tf.random.set_seed(seed + j)
                        tf.keras.utils.set_random_seed(seed + j)

                        model = SOM()
                        alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[map_height, map_width], mem_max=ram, adjust_params=False)
                        acc_, fpr_, tpr_, auc_, preds_ = model.evaluate(data['inf'], alpha, args.level)

                        acc += acc_
                        fpr += fpr_
                        tpr += tpr_
                        auc += auc_
                        preds.extend(preds_)

                    acc /= ntries
                    fpr /= ntries
                    tpr /= ntries
                    auc /= ntries

                except Exception as e:
                    print(e)
                    acc, fpr, tpr, auc = 0.5, 0.0, 0.0, 0.5

                print('params:', config, 'acc:', acc)

            progress.append(f'{acc},{fpr},{tpr},{auc},{config["map_height"]},{config["map_width"]}\n')

            return {"acc": acc, 'fpr': fpr, 'tpr': tpr, 'auc': auc}

        def train_dsvdd_model(config):

            # sampled data

            #n_layers = int(np.round(config['n_layers']))
            n_layers = 1
            n_units = int(np.round(config['n_units']))
            n_outputs = int(np.round(config['n_outputs']))

            layer_sizes = np.array([np.prod(input_shape)] + [n_units for _ in range(n_layers)] + [n_outputs])
            model_size = preproc_size + layer_sizes[-1] * 4 / 1024
            for i in range(len(layer_sizes) - 1):
                model_size += (layer_sizes[i] * layer_sizes[i+1] + 3 * layer_sizes[i+1] + layer_sizes[i]) * 4 / 1024

            print('The model size to the ram ratio:', model_size / ram)

            if model_size > ram:
                #acc, fpr, tpr, auc = acc_baseline, fpr_baseline, tpr_baseline, auc_baseline
                acc, fpr, tpr, auc = 0.5, 0.0, 0.0, 0.5
                print('*********************')
                print('The model is too big!')
                print('*********************')

            else:

                try:

                    acc, fpr, tpr, auc, preds = 0, 0, 0, 0, []

                    for j in range(ntries):

                        np.random.seed(seed + j)
                        tf.random.set_seed(seed + j)
                        tf.keras.utils.set_random_seed(seed + j)

                        model = DSVDD()
                        alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_layers, n_units, n_outputs], mem_max=ram, adjust_params=False)
                        acc_, fpr_, tpr_, auc_, preds_ = model.evaluate(data['inf'], alpha, args.level)

                        acc += acc_
                        fpr += fpr_
                        tpr += tpr_
                        auc += auc_
                        preds.extend(preds_)

                    acc /= ntries
                    fpr /= ntries
                    tpr /= ntries
                    auc /= ntries

                except Exception as e:
                    print(e)
                    acc, fpr, tpr, auc = 0.5, 0.0, 0.0, 0.5

                print('params:', config, 'acc:', acc)

            progress.append(f'{acc},{fpr},{tpr},{auc},{n_layers},{config["n_units"]},{config["n_outputs"]}\n')

            return {"acc": acc, 'fpr': fpr, 'tpr': tpr, 'auc': auc}

        if args.algorithm == 'cs':
            n_clusters_max = 8
            p_microclusters_max = 7
            print('N clusters max:', n_clusters_max, 'p microclusters max:', p_microclusters_max)
            config_search_space = {
                'n_clusters': tune.uniform(2, n_clusters_max),
                'p_microclusters': tune.uniform(1.0, p_microclusters_max),
            }

            trainable = train_cs_model

        elif args.algorithm == 'skm':
            n_clusters_max = 8
            p_batch_size_max = 7
            print('N clusters max:', n_clusters_max, 'p batch size max:', p_batch_size_max)
            config_search_space = {
                'n_clusters': tune.uniform(2, n_clusters_max),
                'p_batch_size': tune.uniform(1.0, p_batch_size_max),
            }

            trainable = train_skm_model

        elif args.algorithm == 'som':
            map_height_max = 10
            map_width_max = 10
            config_search_space = {
                'map_height': tune.uniform(2, map_height_max),
                'map_width': tune.uniform(2, map_width_max),
            }

            trainable = train_som_model

        elif args.algorithm == 'dsvdd':
            n_layers_max = 1
            n = 2
            while (True):
                if preproc_size + (np.prod(input_shape) * n + n + n * n + n + np.prod(input_shape) + n + n + n + n + n + n) * 4 / 1024 > ram:
                    break
                n_units_max = n
                n += 1

            #n_units_max = np.minimum(n_units_max, 117)

            print(f'N units max: {n_units_max}')
            config_search_space = {
                #'n_layers': tune.uniform(1, n_layers_max),
                'n_units': tune.uniform(2, n_units_max),
                'n_outputs': tune.uniform(2, n_units_max),
            }

            trainable = train_dsvdd_model

        else:
            raise NotImplemented

        print(config_search_space)

        #np.random.seed(seed)
        #tf.random.set_seed(seed)
        #tf.keras.utils.set_random_seed(seed)

        tuner = tune.run(
            trainable, metric="acc", mode="max", config=config_search_space, search_alg=BlendSearch(), time_budget_s=None, num_samples=args.trials,
        )

        print('Best config:', tuner.best_config)
        print('Best results:', tuner.best_result)

        print(len(progress))

        with open(osp.join(model_fpath, 'progress.csv'), 'w') as f:
            f.writelines(progress)

        acc, fpr, tpr, auc, preds = 0, 0, 0, 0, []

        ntries = 3
        for j in range(ntries):

            np.random.seed(seed + j)
            tf.random.set_seed(seed + j)
            tf.keras.utils.set_random_seed(seed + j)

            if args.algorithm == 'cs':

                params = tuner.best_config
                print('Params:', params)

                n_clusters = int(np.round(params['n_clusters']))
                p_microclusters = params['p_microclusters']

                model = CS()
                alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_clusters, p_microclusters], mem_max=ram, adjust_params=False)
                #alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_clusters, p_microclusters], mem_max=ram, adjust_params=True)

                #joblib.dump(model, osp.join(model_fpath, 'model.joblib'))

            elif args.algorithm == 'skm':

                params = tuner.best_config
                print('Params:', params)

                n_clusters = int(np.round(params['n_clusters']))
                p_batch_size = params['p_batch_size']

                model = CS()
                alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_clusters, p_batch_size], mem_max=ram, adjust_params=False)
                #alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_clusters, p_batch_size], mem_max=ram, adjust_params=True)

                #joblib.dump(model, osp.join(model_fpath, 'model.joblib'))

            elif args.algorithm == 'som':

                params = tuner.best_config
                print('Params:', params)

                map_width = int(np.round(params['map_width']))
                map_height = int(np.round(params['map_height']))

                model = SOM()
                #alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[map_height, map_width], mem_max=ram, adjust_params=False)
                alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[map_height, map_width], mem_max=ram, adjust_params=True)

                #joblib.dump(model, osp.join(model_fpath, 'model.joblib'))

            elif args.algorithm == 'dsvdd':

                params = tuner.best_config
                print('Params:', params)

                #n_layers = int(np.round(params['n_layers']))
                n_layers = 1
                n_units = int(np.round(params['n_units']))
                n_outputs = int(np.round(params['n_outputs']))

                model = DSVDD()
                alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_layers, n_units, n_outputs], mem_max=ram, adjust_params=False)
                #alpha, metric_val = model.fit(data['tr'], validation_data=data['val'], hp=[n_layers, n_units], mem_max=ram, adjust_params=True)

                #joblib.dump(model, osp.join(model_fpath, 'model.joblib'))

            else:
                raise NotImplemented

            print(f'Saved to {model_fpath}')

            acc_, fpr_, tpr_, auc_, preds_ = model.evaluate(data['inf'], alpha, args.level)

            acc += acc_
            fpr += fpr_
            tpr += tpr_
            auc += auc_
            preds.extend(preds_)

        acc /= ntries
        fpr /= ntries
        tpr /= ntries
        auc /= ntries

        print(f'\nAcc: {acc}, FPR: {fpr}, TPR: {tpr}, AUC: {auc}\n')

        stats = {
            'acc': acc,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc,
        }

        stats.update(params)

        with open(osp.join(model_fpath, 'stats.json'), 'w') as f:
            json.dump(stats, f)