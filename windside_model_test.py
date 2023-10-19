import json, joblib, os

import argparse as arp
import tensorflow as tf
import numpy as np

from config import *
from windside_extract_data import create_dataset, split_dataset

from matplotlib import pyplot as pp
from scipy.stats import binned_statistic

if __name__ == '__main__':

    #weights = 'continuous'
    weights = 'discrete'

    parser = arp.ArgumentParser(description='Test windisde models.')
    parser.add_argument('-l', '--label', help='Label', default='mean', choices=['last', 'mean'])
    parser.add_argument('-z', '--zeros', help='Filter zeros?', default='filter', choices=['filter', 'only'])
    parser.add_argument('-c', '--chunk_size', help='Chunk size', type=int, default=None)
    parser.add_argument('-m', '--models', help='Model names', default=[
        #'tf_fd2626b2-4e2a-3808-b1cd-cf17031a2017_filter-zeros_mean_raw_weighted',
        #f'tf_c4d4b371-9b8e-3b94-a86b-30f398be7e85_filter-zeros_mean_raw_{weights}',
        'lora_dnn_auto_ray',
        'lora_dnn_auto_ray_weighted_gpuserv',
        #'lora_dnn_auto_ray_weighted_0.05',
        #'lora_dnn_auto_ray_weighted_continuous',
        'lora_dnn_auto_ray_weighted_last'
        #'lora_gbdt_auto_ray'
    ])
    parser.add_argument('-w', '--weights', help='Weight type', default=weights)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    data_fpath = osp.join(DATA_DIR, 'windside', 'features.csv')
    dataset_fpath_pattern = osp.join(DATA_DIR, 'windside', '{0}.{1}')

    feature_cols = ['Current']

    series_step = 1
    window_size = 60

    X, T, Y, D, W = create_dataset(
        data_fpath,
        output_fpath_pattern=dataset_fpath_pattern,
        feature_cols=feature_cols,
        series_step=series_step,
        window_size=window_size,
        window_step=1,
        chunk_size=args.chunk_size,
        label_mode=args.label,
        feature_extractor='raw',
        weights=args.weights
    )

    X_tr, T_tr, Y_tr, D_tr, W_tr, X_val, T_val, Y_val, D_val, W_val, X_inf, T_inf, Y_inf, D_inf, W_inf, y_tr_0_mean, y_tr_1_mean = split_dataset(
        (X, T, Y, D, W),
        zeros=args.zeros,
        inf_split=0.1,
        shuffle_all=False,
        shuffle_train=False
    )

    print('Data has been loaded!')

    task_model_dir = WINDSIDE_MODEL_DIR
    model_names = args.models
    model_fpaths = [osp.join(task_model_dir, model_name) for model_name in model_names]
    meta_fpaths = [osp.join(model_fpath, 'metainfo.json') for model_fpath in model_fpaths]

    model_labels = [
        'Baseline (no weights)',
        'Histogram based weights',
        'Density based weights'
    ]

    model, model_type = None, None
    errors = []

    X = np.vstack([X_tr, X_val, X_inf])
    T = np.vstack([T_tr, T_val, T_inf])
    Y = np.hstack([Y_tr, Y_val, Y_inf])
    W = np.hstack([W_tr, W_val, W_inf])

    X_mean = np.mean(X.squeeze(), 1)

    idx_sorted = np.argsort(X_mean)

    fig, ax = pp.subplots(1)
    fig.set_size_inches(5, 5)
    ax.plot(X_mean[idx_sorted], W[idx_sorted], 'ro')
    pp.xlabel('Mean current, A')
    pp.ylabel('Weight value')
    pp.savefig('weights_.png', bbox_inches='tight', pad_inches=0)
    pp.close()

    bin_step = 0.1
    n_bins = (np.max(X_mean) // bin_step) + 1
    print(n_bins)

    idx_sorted = np.argsort(X_mean)

    k = 2

    X_k = X * k
    X_k_mean = np.mean(X_k.squeeze(), 1)

    n_samples = X.shape[0]

    markers = []
    labels = []

    M = ['bo-', 'rs-', 'm^-', 'gx-']

    count = 0

    for model_name, model_fpath, meta_fpath, model_label in zip(model_names, model_fpaths, meta_fpaths, model_labels):

        try:
            model = tf.keras.models.load_model(model_fpath, compile=False)
            model_type = 'tf'
            print(f'Loaded TF model: {model_name}')
        except Exception as e:
            print(e)
            pass

        try:
            model = joblib.load(osp.join(model_fpath, 'model.joblib'))
            print(model.get_params())
            model_type = 'skl'
            print(f'Loaded Skl model: {model_name}')
        except:
            pass

        if model is not None:

            with open(meta_fpath) as f:
                metadata = json.load(f)

            P_all = model.predict(
                #[np.vstack([X, X_k]), np.vstack([T, T])],
                np.vstack([X, X_k]).squeeze(),
                batch_size=2048
            ).flatten()
            P = P_all[:n_samples]
            P_k = P_all[n_samples:]

            E = np.abs(Y - P)

            errors.append(E)
            markers.append(M[count])
            labels.append(model_label)

            count += 1

            fig, ax = pp.subplots(1)
            fig.set_size_inches(12, 12)

            ax.plot(X_mean[idx_sorted], Y[idx_sorted], 'go', label=f'Real wind speed')
            ax.plot(X_k_mean[idx_sorted], P_k[idx_sorted], 'rx', label=f'Predicted wind speed for current x{k}')
            ax.plot(X_mean[idx_sorted], P[idx_sorted], 'bo', label=f'Predicted wind speed for original current')

            pp.legend()
            pp.xlabel('Mean current, A')
            pp.ylabel('Wind speed, m/s')

            pp.savefig(f'{model_name}_current_x{k}.png', bbox_inches='tight', pad_inches=0)
            pp.close(fig)

        else:
            print('Model should be either created in Tensorflow or in one of the following packages: Sklearn, Lightning, XGBoost or LightGBM!')
            raise NotImplemented

    fig, ax = pp.subplots(1)
    fig.set_size_inches(12, 5)

    for L, M, E in zip(labels, markers, errors):
        h, e, _ = binned_statistic(X_mean[idx_sorted], E[idx_sorted], bins=n_bins)
        #ax.plot(X_mean[idx_sorted], E[idx_sorted], M, label=L)
        ax.plot(e[1:], h, M, label=L)

    pp.legend()
    pp.xlabel('Mean current, A')
    pp.ylabel('Mean absolute error, m/s')

    pp.savefig(f'error_{args.weights}.png', bbox_inches='tight', pad_inches=0)