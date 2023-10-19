import json, os, uuid, joblib
import pandas as pd
import numpy as np
import argparse as arp

from time import time
from config import *
from windside_extract_data import create_dataset, split_dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import ElasticNet as ENR
from sklearn.linear_model import SGDRegressor as SGDR
from sklearn.linear_model import BayesianRidge as BRR
from sklearn.svm import SVR
from lightning.regression import LinearSVR as LSVR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from xgboost.sklearn import XGBRFRegressor as XGBRFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from xgboost.sklearn import XGBRegressor as XGBR
from lightgbm.sklearn import LGBMRegressor as LGBMR


if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Train wind speed predictor.')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=42)
    parser.add_argument('-c', '--chunk_size', help='Chunk size', type=int, default=None)
    parser.add_argument('-o', '--order', help='Sorting order', default='date', choices=['date', 'windspeed', 'random'])
    parser.add_argument('-l', '--label_mode', help='Label mode', default='avg+max', choices=['avg', 'max', 'avg+max'])
    parser.add_argument('-a', '--algorithms', help='Algorithms', nargs='+', default=['LR', 'XGBRFR', 'XGBR'], choices=['LR', 'ENR', 'SGDR', 'BRR', 'SVR', 'LSVR', 'DTR', 'RFR', 'XGBRFR', 'GBR', 'XGBR', 'LGBMR'])
    args = parser.parse_args()

    include_voltage = False
    seed = args.seed

    series_step = 1
    window_size = 60

    val_split = 0.2
    inf_split = 0.2

    np.random.seed(seed)

    data_fpath = osp.join(DATA_DIR, 'windside', 'features.csv')
    dataset_fpath_pattern = osp.join(DATA_DIR, 'windside', '{0}.{1}')

    feature_cols = ['Current']
    if include_voltage:
        feature_cols.append('Voltage')

    X, T, Y, D, W, y0_mean, y0_error = create_dataset(
        data_fpath,
        output_fpath_pattern=dataset_fpath_pattern,
        feature_cols=feature_cols,
        series_step=series_step,
        window_size=window_size,
        window_step=1,
        chunk_size=args.chunk_size,
        label_mode=args.label_mode
    )

    print(X.shape, Y.shape, W.shape)

    X_tr, T_tr, Y_tr, D_tr, W_tr, X_val, T_val, Y_val, D_val, W_val, X_inf, T_inf, Y_inf, D_inf, W_inf = split_dataset(
        (X, T, Y, D, W),
        inf_split=inf_split,
        order=args.order,
        shuffle_train=False
    )

    X_tr = np.squeeze(X_tr)
    X_val = np.squeeze(X_val)
    X_inf = np.squeeze(X_inf)

    x_mean = np.mean(X_tr)
    x_std = np.std(X_tr)

    # scaler = MinMaxScaler()
    scaler = StandardScaler()

    X_tr_std = scaler.fit_transform(X_tr)
    X_val_std = scaler.transform(X_val)
    X_inf_std = scaler.transform(X_inf)

    print('The training set dates:', D_tr[0], D_tr[-1])
    print('The validation set dates:', D_val[0], D_val[-1])
    if len(D_inf) > 0:
        print('The inference set dates:', D_inf[0], D_inf[-1])

    for algorithm in args.algorithms:

        np.random.seed(seed)

        t_start = time()

        print(f'\nTrying {algorithm}...\n')

        if algorithm == 'SVR':
            kwargs = {'kernel': 'rbf', 'gamma': 'auto', 'max_iter': 512}
        elif algorithm == 'DTR':
            kwargs = {'random_state': args.seed}
        elif algorithm in ['RFR', 'XGBRFR', 'GBR', 'XGBR', 'LGBMR']:
            kwargs = {'random_state': args.seed, 'n_estimators': 64, 'num_leaves': 64}
        else:
            kwargs = {}

        if algorithm == 'XGBR':
            kwargs['verbosity'] = 2
        elif algorithm not in ['LR', 'ENR', 'DTR']:
            kwargs['verbose'] = 1

        model_fname = f'{algorithm.lower()}_{uuid.uuid3(uuid.NAMESPACE_DNS, json.dumps(kwargs))}_{args.order}_{args.label_mode}'
        #model_fname = f'{algorithm.lower()}_{uuid.uuid3(uuid.NAMESPACE_DNS, json.dumps(kwargs))}_{mode if mode is not None else "all-samples"}_{args.label}'
        # model_fname = f'{uuid.uuid3(uuid.NAMESPACE_DNS, json.dumps(args.algorithm))}_{mode if mode is not None else "all-samples"}'
        # model_fname = f'{algorithm.lower()}_{mode if mode is not None else "all-samples"}'
        model_fpath = osp.join(WINDSIDE_MODEL_DIR, model_fname)
        print('Model fpath:', model_fpath)

        model = globals()[algorithm](**kwargs)

        model.fit(
            X_tr_std,
            Y_tr
        )

        # quick test

        print(f'Quick test result: {model.predict(X_tr_std[[0]])}')

        if not osp.isdir(model_fpath):
            os.mkdir(model_fpath)
        joblib.dump(model, osp.join(model_fpath, 'model.joblib'))
        print(f'Saved to {model_fpath}')

        with open(osp.join(model_fpath, 'metainfo.json'), 'w') as f:
            json.dump({
                'input_shape': X_tr.shape,
                'output_shape': Y_tr.shape if len(Y_tr.shape) > 1 else (Y_tr.shape[0], 1),
                'x_mean': float(x_mean),
                'x_std': float(x_std),
                'y0_avg_mean': float(y0_mean[0]),
                'y0_max_mean': float(y0_mean[1]),
                'y0_avg_error': float(y0_error[0]),
                'y0_max_error': float(y0_error[1])
            }, f)

        if X_inf.shape[0] > 0:

            X_inf_sum = np.sum(X_inf, 1)

            idx_inf_1 = np.where(X_inf_sum > 0)[0]
            print('Number of positive current samples in the inference subset =', len(idx_inf_1))
            Y1 = Y_inf[idx_inf_1]

            P1 = model.predict(X_inf_std[idx_inf_1, :])

            error1_arr = np.abs(Y1 - P1)
            idx_avg = np.argsort(Y1[:, 0])
            idx_max = np.argsort(Y1[:, 1])
            error1_arr_sorted = np.hstack([
                Y1[idx_avg, 0].reshape(-1, 1),
                error1_arr[idx_avg, 0].reshape(-1, 1),
                Y1[idx_max, 1].reshape(-1, 1),
                error1_arr[idx_max, 1].reshape(-1, 1)
            ])

            error1 = np.mean(error1_arr, axis=0)
            error1_std = np.std(np.abs(Y1 - P1), axis=0)

            print(f'\nError 0: {y0_error}, error 1: {error1}')

            with open(osp.join(model_fpath, 'stats.json'), 'w') as f:
                json.dump({
                    'error0': y0_error.tolist(),
                    'error1': error1.tolist(),
                    'model': json.dumps(algorithm)
                }, f)

            pd.DataFrame(error1_arr_sorted).to_csv(osp.join(model_fpath, 'error.csv'), header=None, index=None)

        t_end = time()

        print(f'Completed in {(t_end - t_start) / 60} minutes!')