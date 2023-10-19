import numpy as np
import json, lightgbm, joblib
import argparse as arp
import tensorflow as tf

from config import *
from windside_extract_data import create_dataset, split_dataset, sample_data
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from flaml.automl.model import LGBMEstimator
from ray import tune, air
from ray.tune.search.bayesopt import BayesOptSearch
from utils.boards import get_gbdt_model_limits, get_dnn_model_limits, get_dt_model_size, get_ram_flash
from io import BytesIO

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Train auto (Bayes search) regression model to estimate wind speed.')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=42)
    parser.add_argument('-l', '--label', help='Label', default='mean', choices=['last', 'mean'])
    parser.add_argument('-z', '--zeros', help='Filter zeros?', default='filter', choices=['filter', 'only'])
    parser.add_argument('-c', '--chunk_size', help='Chunk size', type=int, default=None)
    parser.add_argument('-n', '--ntrain', help='Number of training samples', type=int)
    parser.add_argument('-r', '--sampling_rate', help='Sampling rate', type=float, default=0.2)
    parser.add_argument('-v', '--verbose', help='Verbosity', type=int, default=0)
    parser.add_argument('-t', '--trials', help='Number of trials', type=int, default=100)
    parser.add_argument('-b', '--boards', help='Board names', nargs='+', default=['lora'])
    parser.add_argument('-a', '--algorithm', help='Algorithm', default='dnn', choices=['gbdt', 'dnn'])
    parser.add_argument('-p', '--postfix', help='Postfix in the name', default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.zeros == 'filter':
        mode = 'filter-zeros'
    elif args.zeros == 'only':
        mode = 'only-zeros'
    else:
        mode = None

    seed = args.seed
    np.random.seed(seed)

    val_split = 0.2
    output_dim = 1
    loss = 'mean_squared_error'
    epochs = 1000
    patience = 30
    lr = 1e-3
    batch_size = 4096
    dropout = 0.5
    metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

    data_fpath = osp.join(DATA_DIR, 'windside', 'features.csv')
    dataset_fpath_pattern = osp.join(DATA_DIR, 'windside', '{0}.{1}')

    t_start = time()
    X, T, Y, D, W = create_dataset(
        data_fpath,
        #output_fpath_pattern=dataset_fpath_pattern,
        feature_cols=['Current'],
        series_step=1,
        window_size=60,
        window_step=1,
        chunk_size=args.chunk_size,
        label_mode=args.label,
    )
    print(f'Created in {time() - t_start} seconds')

    t_start = time()
    X_tr, T_tr, Y_tr, D_tr, W_tr, X_val, T_val, Y_val, D_val, W_val, X_inf, T_inf, Y_inf, D_inf, W_inf, y_tr_0_mean, y_tr_1_mean = split_dataset(
        (X, T, Y, D, W),
        zeros=args.zeros,
        val_split=val_split,
        inf_split=0.1,
        shuffle_all=False,
        shuffle_train=False,
    )
    print(f'Split in {time() - t_start} seconds')

    print('Zero current mean wind speed value:', y_tr_0_mean)

    X_tr = X_tr.squeeze()
    X_val = X_val.squeeze()
    X_inf = X_inf.squeeze()

    print('Original shapes: ', X_tr.shape, Y_tr.shape, X_val.shape, Y_val.shape, X_inf.shape, Y_inf.shape)

    X_tr_, T_tr_, Y_tr_, D_tr_, W_tr_ = sample_data(X_tr, T_tr, Y_tr, D_tr, W_tr, sampling_rate=args.sampling_rate)
    X_val_, T_val_, Y_val_, D_val_, W_val_ = sample_data(X_val, T_val, Y_val, D_val, W_val, sampling_rate=args.sampling_rate)

    print('Shapes after resampling: ', X_tr_.shape, Y_tr_.shape, X_val_.shape, Y_val_.shape, X_inf.shape, Y_inf.shape)

    x_mean = np.mean(X_tr)
    x_std = np.std(X_tr)
    y_min = np.min(Y_tr)
    y_max = np.max(Y_tr)

    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr)
    X_tr_std_ = scaler.transform(X_tr_)
    X_val_std = scaler.transform(X_val)
    X_val_std_ = scaler.transform(X_val_)
    X_inf_std = scaler.transform(X_inf)

    input_shape = X_tr.shape[1:]
    output_dim = 1

    train_set_ = lightgbm.Dataset(data=X_tr_std_, label=Y_tr_)
    train_set = lightgbm.Dataset(data=X_tr_std, label=Y_tr)

    X_inf_sum = np.array([np.sum(x) for x in X_inf])
    X_tr_shape = X_tr.shape
    x_test = X_tr[[0]].flatten().tolist()

    del X_tr
    del X_val

    # baseline model

    model_baseline_ = LinearRegression()
    model_baseline_.fit(X_tr_std_, Y_tr_)
    pred = model_baseline_.predict(X_val_std_)
    mae_baseline = np.mean(np.abs(Y_val_ - pred))

    model_baseline = LinearRegression()
    model_baseline.fit(X_tr_std, Y_tr)

    idx_inf_0 = np.where(X_inf_sum == 0)[0]
    idx_inf_1 = np.where(X_inf_sum > 0)[0]
    Y0 = Y_inf[idx_inf_0]
    Y1 = Y_inf[idx_inf_1]

    if mode == 'baseline':
        P0 = np.ones(len(idx_inf_0)) * y_tr_0_mean
        P1 = np.ones(len(idx_inf_1)) * y_tr_1_mean
    elif mode == 'filter-zeros':
        P0 = np.ones(len(idx_inf_0)) * y_tr_0_mean
        P1 = model_baseline.predict(
            X_inf_std[idx_inf_1, :],
        )
    elif mode == 'only-zeros':
        P0 = model_baseline.predict(
            X_inf_std[idx_inf_0, :],
        )
        P1 = np.ones(len(idx_inf_1)) * y_tr_1_mean
    else:
        P0 = model_baseline.predict(
            X_inf_std[idx_inf_0, :],
        )
        P1 = model_baseline.predict(
            X_inf_std[idx_inf_1, :],
        )

    error0_baseline = np.mean(np.abs(Y0 - P0))
    error1_baseline = np.mean(np.abs(Y1 - P1))
    error_baseline = (error0_baseline * len(Y0) + error1_baseline * len(Y1)) / len(Y_inf)

    print('MAE baseline:', mae_baseline, 'Error0 baseline:', error0_baseline, 'Error1 baseline:', error1_baseline, 'Error baseline:', error1_baseline)

    baseline_model_fname = f'baseline_lr'
    baseline_model_fpath = osp.join(WINDSIDE_MODEL_DIR, baseline_model_fname)
    print('Baseline model fpath:', baseline_model_fpath)

    if not osp.isdir(baseline_model_fpath):
        os.mkdir(baseline_model_fpath)

    with open(osp.join(baseline_model_fpath, 'metainfo.json'), 'w') as f:
        json.dump({
            'input_shape': X_tr_shape,
            'output_shape': Y_tr.shape if len(Y_tr.shape) > 1 else (Y_tr.shape[0], 1),
            'x_mean': x_mean,
            'x_std': x_std,
            'y_min': y_min,
            'y_max': y_max,
            'y0_mean': y_tr_0_mean,
            'x_test': x_test
        }, f)

    with open(osp.join(baseline_model_fpath, 'stats.json'), 'w') as f:
        json.dump({
            'mae': mae_baseline,
            'error0': error0_baseline,
            'error1': error1_baseline,
            'error': error_baseline
        }, f)

    for board in args.boards:

        #model_fname = f'{board}_{args.algorithm}_auto_{mode if mode is not None else "all-samples"}_{args.label}'
        model_fname = f'{board}_{args.algorithm}_auto_ray_weighted'
        if args.postfix is not None:
            model_fname = f'{model_fname}_{args.postfix}'
        model_fpath = osp.join(WINDSIDE_MODEL_DIR, model_fname)
        print('Model fpath:', model_fpath)

        if not osp.isdir(model_fpath):
            os.mkdir(model_fpath)

        with open(osp.join(model_fpath, 'metainfo.json'), 'w') as f:
            json.dump({
                'input_shape': X_tr_shape,
                'output_shape': Y_tr.shape if len(Y_tr.shape) > 1 else (Y_tr.shape[0], 1),
                'x_mean': x_mean,
                'x_std': x_std,
                'y_min': y_min,
                'y_max': y_max,
                'y0_mean': y_tr_0_mean,
                'x_test': x_test
            }, f)

        # main part

        dt_k, dt_b = get_dt_model_size(board=board)
        ram, flash = get_ram_flash(board=board)

        def train_gbdt_model(config, data=None):

            # sampled data

            for key in config.keys():
                config[key] = int(np.round(config[key]))

            config['random_state'] = args.seed

            params = LGBMEstimator(**config).params

            model_ = lightgbm.train(params=params, train_set=data['tr'][0])

            with BytesIO() as tmp_bytes:
                joblib.dump(model_, tmp_bytes)
                model_size = tmp_bytes.getbuffer().nbytes / 1024

            model_size_on_board = dt_k * model_size + dt_b
            if model_size_on_board > flash:
                print('The model is too big!')

                mae = mae_baseline
                error0 = error0_baseline
                error1 = error1_baseline
                error = error_baseline

            else:

                pred = model_.predict(data['val'][0])
                mae = np.mean(np.abs(data['val'][1] - pred))

                error0 = error0_baseline
                error1 = error1_baseline
                error = error_baseline

                # full data

                if 0:

                    model = lightgbm.train(params=params, train_set=data['tr'][1])

                    X_inf_sum = data['inf'][2]
                    idx_inf_0 = np.where(X_inf_sum == 0)[0]
                    idx_inf_1 = np.where(X_inf_sum > 0)[0]
                    Y0 = data['inf'][1][idx_inf_0]
                    Y1 = data['inf'][1][idx_inf_1]

                    if mode == 'baseline':
                        P0 = np.ones(len(idx_inf_0)) * y_tr_0_mean
                        P1 = np.ones(len(idx_inf_1)) * y_tr_1_mean
                    elif mode == 'filter-zeros':
                        P0 = np.ones(len(idx_inf_0)) * y_tr_0_mean
                        P1 = model.predict(
                            data['inf'][0][idx_inf_1, :],
                        )
                    elif mode == 'only-zeros':
                        P0 = model.predict(
                            data['inf'][0][idx_inf_0, :],
                        )
                        P1 = np.ones(len(idx_inf_1)) * y_tr_1_mean
                    else:
                        P0 = model.predict(
                            data['inf'][0][idx_inf_0, :],
                        )
                        P1 = model.predict(
                            data['inf'][0][idx_inf_1, :],
                        )

                    error0 = np.mean(np.abs(Y0 - P0))
                    error1 = np.mean(np.abs(Y1 - P1))
                    error = (error0 * len(Y0) + error1 * len(Y1)) / len(Y_inf)

            print('params:', params, 'mae:', mae, ', error1:', error1)

            return {"mae": mae, 'error0': error0, 'error1': error1, 'error': error}


        def train_dnn_model(config, data=None):

            # sampled data

            for key in config.keys():
                config[key] = int(np.round(config[key]))

            n_layers = config['n_layers']
            n_units = config['n_units']

            inputs = tf.keras.layers.Input(shape=input_shape)

            hidden = inputs
            model_size_on_board = 0
            for i in range(n_layers):
                hidden = tf.keras.layers.Dense(units=n_units, activation='relu')(hidden)
                hidden = tf.keras.layers.Dropout(dropout)(hidden)
                if i == 0:
                    model_size_on_board += (np.prod(input_shape) * n_units + n_units)
                else:
                    model_size_on_board += (n_units * n_units + n_units)

            outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(hidden)

            model_size_on_board += (n_units * output_dim + output_dim)

            model_ = tf.keras.models.Model(inputs=inputs, outputs=outputs)

            model_.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), weighted_metrics=metrics)

            print(data['tr'][2].shape)
            print(data['val'][4].shape)

            model_.fit(
                data['tr'][0].data, data['tr'][0].label,
                sample_weight=data['tr'][2],
                validation_data=(data['val'][0], data['val'][1], data['val'][4]),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
                ],
                verbose=False
            )

            model_size_on_board = model_size_on_board * 4 / 1024

            print('The model size to the flash ratio: (', model_size_on_board, '/', flash, ') =', model_size_on_board / flash)

            if model_size_on_board > flash:
                print('*********************')
                print('The model is too big!')
                print('*********************')

                mae = mae_baseline
                error0 = error0_baseline
                error1 = error1_baseline
                error = error_baseline

            else:

                pred = model_.predict(data['val'][0], verbose=False).reshape(1, -1)
                mae = np.mean(np.abs(data['val'][1] - pred))

                error0 = error0_baseline
                error1 = error1_baseline
                error = error_baseline

                # full data

                if 0:

                    model = tf.keras.models.clone_model(model_)
                    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=metrics)
                    model.fit(
                        data['tr'][1].data, data['tr'][1].label,
                        validation_data=(data['val'][2], data['val'][3]),
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
                        ],
                        verbose=False
                    )

                    X_inf_sum = data['inf'][2]
                    idx_inf_0 = np.where(X_inf_sum == 0)[0]
                    idx_inf_1 = np.where(X_inf_sum > 0)[0]
                    Y0 = data['inf'][1][idx_inf_0]
                    Y1 = data['inf'][1][idx_inf_1]

                    if mode == 'baseline':
                        P0 = np.ones(len(idx_inf_0)) * y_tr_0_mean
                        P1 = np.ones(len(idx_inf_1)) * y_tr_1_mean
                    elif mode == 'filter-zeros':
                        P0 = np.ones(len(idx_inf_0)) * y_tr_0_mean
                        P1 = model.predict(
                            data['inf'][0][idx_inf_1, :], verbose=False
                        ).reshape(1, -1)
                    elif mode == 'only-zeros':
                        P0 = model.predict(
                            data['inf'][0][idx_inf_0, :], verbose=False
                        ).reshape(1, -1)
                        P1 = np.ones(len(idx_inf_1)) * y_tr_1_mean
                    else:
                        P0 = model.predict(
                            data['inf'][0][idx_inf_0, :], verbose=False
                        ).reshape(1, -1)
                        P1 = model.predict(
                            data['inf'][0][idx_inf_1, :], verbose=False
                        ).reshape(1, -1)

                    error0 = np.mean(np.abs(Y0 - P0))
                    error1 = np.mean(np.abs(Y1 - P1))
                    error = (error0 * len(Y0) + error1 * len(Y1)) / len(Y_inf)

            print('params:', config, 'mae:', mae)

            #with open(osp.join(PROJECT_DIR, model_fpath, 'progress.csv'), 'a') as f:
            #    f.write(f'{error0},{error1},{error},{config["n_estimators"]},{config["num_leaves"]}\n')

            return {"mae": mae, 'error0': error0, 'error1': error1, 'error': error}

        if args.algorithm == 'gbdt':
            n_estimators_max, num_leaves_max = get_gbdt_model_limits(board=board)
            print('N estimators max:', n_estimators_max, 'Num leaves max:', num_leaves_max)
            config_search_space = {
                'n_estimators': tune.uniform(1, n_estimators_max),
                'num_leaves': tune.uniform(2, num_leaves_max),
            }

            trainable = train_gbdt_model

        elif args.algorithm == 'dnn':
            n_layers_max, n_units_max = get_dnn_model_limits(board=board)
            print('N layers max:', n_layers_max, 'n units max:', n_units_max)
            config_search_space = {
                'n_layers': tune.uniform(1, n_layers_max),
                'n_units': tune.uniform(1, n_units_max),
            }

            trainable = train_dnn_model

        tuner = tune.Tuner(
            tune.with_parameters(trainable, data={
                'tr': [train_set_, train_set, W_tr_, W_tr],
                'val': [X_val_std_, Y_val_, X_val_std, Y_val, W_val_, W_val],
                'inf': [X_inf_std, Y_inf, X_inf_sum]
            }),
            param_space=config_search_space,
            tune_config=tune.TuneConfig(
                search_alg=BayesOptSearch(),
                metric="mae", mode="min",
                num_samples=args.trials,
                max_concurrent_trials=1
            ),
            run_config=air.config.RunConfig(verbose=args.verbose)
        )
        tuner.fit()

        best_config = tuner.get_results().get_best_result().config
        print('Best results:', tuner.get_results().get_best_result().metrics)

        df = tuner.get_results().get_dataframe()
        maes = df['mae'].values
        errors0 = df['error0'].values
        errors1 = df['error1'].values
        errors = df['error'].values

        if args.algorithm == 'gbdt':
            hp_keys = ['config/n_estimators', 'config/num_leaves']
        elif args.algorithm == 'dnn':
            hp_keys = ['config/n_layers', 'config/n_units']

        hps = df[hp_keys].values

        print(len(maes), len(errors0), len(errors1), len(errors), len(hps))

        #maes_, errors0_, errors1_, errors_, n_estimators_, num_leaves_ = [], [], [], [], [], []
        #for i in range(len(errors)):
        #    #idx = np.argmin(maes[: i + 1])
        #    idx = np.argmin(errors[: i + 1])
        #    maes_.append(maes[idx])
        #    errors0_.append(errors0[idx])
        #    errors1_.append(errors1[idx])
        #    errors_.append(errors[idx])
        #    n_estimators_.append(n_estimators[idx])
        #    num_leaves_.append(num_leaves[idx])

        #print(len(maes_), len(errors0_), len(errors1_), len(errors_), len(n_estimators_), len(num_leaves_))

        #errors0 = [x for _, x in sorted(zip(maes, errors0), reverse=True)]
        #errors1 = [x for _, x in sorted(zip(maes, errors1), reverse=True)]
        #errors = [x for _, x in sorted(zip(maes, errors), reverse=True)]
        #n_estimators = [x for _, x in sorted(zip(maes, n_estimators), reverse=True)]
        #num_leaves = [x for _, x in sorted(zip(maes, num_leaves), reverse=True)]

        progress = []
        #for error0, error1, error, n_e, n_l in zip(errors0_, errors1_, errors_, n_estimators_, num_leaves_):
        for i, (mae, error0, error1, error) in enumerate(zip(maes, errors0, errors1, errors)):
            hp_line = ','.join([str(hp) for hp in hps[i, :]])
            progress.append(f'{mae},{error0},{error1},{error},{hp_line}\n')

        print(len(progress))

        with open(osp.join(model_fpath, 'progress.csv'), 'w') as f:
            f.writelines(progress)

        if args.algorithm == 'gbdt':

            params = LGBMEstimator(**best_config).params
            model = lightgbm.sklearn.LGBMRegressor(**params)
            model.fit(X_tr_std, Y_tr)
            joblib.dump(model, osp.join(model_fpath, 'model.joblib'))

        elif args.algorithm == 'dnn':

            params = best_config
            inputs = tf.keras.layers.Input(shape=input_shape)
            hidden = inputs
            for i in range(params['n_layers']):
                hidden = tf.keras.layers.Dense(units=params['n_units'], activation='relu')(hidden)
                hidden = tf.keras.layers.Dropout(dropout)(hidden)
            outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(hidden)
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=metrics)
            model.fit(
                X_tr_std, Y_tr,
                sample_weight=W_tr,
                validation_data=(X_val_std, Y_val, W_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
                ],
                verbose=False
            )
            model.save(model_fpath)

        print('Params:', params)
        print(f'Saved to {model_fpath}')

        if X_inf.shape[0] > 0:

            X_inf_sum = np.array([np.sum(x) for x in X_inf])
            idx_inf_0 = np.where(X_inf_sum == 0)[0]
            idx_inf_1 = np.where(X_inf_sum > 0)[0]
            print(len(idx_inf_0), len(idx_inf_1))
            Y0 = Y_inf[idx_inf_0]
            Y1 = Y_inf[idx_inf_1]

            if mode == 'baseline':
                P0 = np.ones(len(idx_inf_0)) * y_tr_0_mean
                P1 = np.ones(len(idx_inf_1)) * y_tr_1_mean
            elif mode == 'filter-zeros':
                P0 = np.ones(len(idx_inf_0)) * y_tr_0_mean
                P1 = model.predict(
                    X_inf_std[idx_inf_1, :],
                ).reshape(1, -1)
            elif mode == 'only-zeros':
                P0 = model.predict(
                    X_inf_std[idx_inf_0, :],
                ).reshape(1, -1)
                P1 = np.ones(len(idx_inf_1)) * y_tr_1_mean
            else:
                P0 = model.predict(
                    X_inf_std[idx_inf_0, :],
                ).reshape(1, -1)
                P1 = model.predict(
                    X_inf_std[idx_inf_1, :],
                ).reshape(1, -1)

            error0 = np.mean(np.abs(Y0 - P0))
            error1 = np.mean(np.abs(Y1 - P1))
            error = (error0 * len(Y0) + error1 * len(Y1)) / len(Y_inf)

            # print(len(Y0), len(Y1))
            print(f'\nError 0: {error0}, error 1: {error1}, error: {error}')

            with open(osp.join(model_fpath, 'stats.json'), 'w') as f:
                json.dump({
                    'error0': error0,
                    'error1': error1,
                    'error': error
                }, f)