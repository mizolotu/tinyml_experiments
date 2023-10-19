import numpy as np
import pandas as pd
import json, lightgbm, joblib
import argparse as arp
import tensorflow as tf

from config import *
from windside_extract_data import create_dataset, split_dataset, sample_data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from flaml import BlendSearch
from flaml import tune as tune_flaml
from flaml.automl.model import LGBMEstimator
from ray import air
from ray import tune as tune_ray
from ray.tune.search.bayesopt import BayesOptSearch
from utils.boards import get_gbdt_model_limits, get_dnn_model_limits, get_cnn_model_limits, get_dt_model_size, get_ram_flash
from io import BytesIO

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Train auto (Bayes search) regression model to estimate wind speed.')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=42)
    parser.add_argument('-f', '--framework', help='AutoML framework', default='flaml', choices=['flaml', 'ray'])
    parser.add_argument('-c', '--chunk_size', help='Chunk size', type=int, default=None)
    parser.add_argument('-o', '--order', help='Order', default='date', choices=['random', 'date', 'windspeed'])
    parser.add_argument('-l', '--label_mode', help='Label mode', default='avg+max', choices=['avg', 'max', 'avg+max'])
    parser.add_argument('-r', '--sampling_rate', help='Sampling rate', type=float, default=0.1)
    parser.add_argument('-v', '--verbose', help='Verbosity', type=int, default=0)
    parser.add_argument('-t', '--trials', help='Number of trials', type=int, default=100)
    parser.add_argument('-b', '--boards', help='Board names', nargs='+', default=['sense', 'iot', 'every', 'uno', 'lora'])
    parser.add_argument('-m', '--memory', help='Memory limit', default='ram', choices=['ram', 'flash'])
    parser.add_argument('-a', '--algorithm', help='Algorithm', default='cnn', choices=['gbdt', 'dnn', 'cnn'])
    parser.add_argument('-w', '--weights', help='Weight type', default=None, choices=[None, 'discrete', 'continuous'])
    parser.add_argument('-g', '--gpu', help='GPU', default='-1')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    series_step = 1
    window_size = 60
    include_voltage = False

    seed = args.seed

    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    inf_split = 0.2
    val_split = 0.2

    error1_too_big = 2.0

    output_dim = 1
    loss = 'mean_squared_error'
    epochs = 1000
    patience = 10
    lr = 1e-3
    batch_size = 4096

    metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

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
        weights=args.weights,
        label_mode=args.label_mode
    )

    X_tr, T_tr, Y_tr, D_tr, W_tr, X_val, T_val, Y_val, D_val, W_val, X_inf, T_inf, Y_inf, D_inf, W_inf = split_dataset(
        (X, T, Y, D, W),
        inf_split=inf_split,
        order=args.order,
        shuffle_train=False
    )

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
    output_dim = 2

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
    mae_baseline = np.mean(np.abs(Y_val_ - pred), 0)

    model_baseline = LinearRegression()
    model_baseline.fit(X_tr_std, Y_tr)

    idx_inf_0 = np.where(X_inf_sum == 0)[0]
    idx_inf_1 = np.where(X_inf_sum > 0)[0]
    Y0 = Y_inf[idx_inf_0, :]
    Y1 = Y_inf[idx_inf_1, :]

    print('Number of zero-current samples:', len(idx_inf_0), 'number of positive current samples:', len(idx_inf_1))
    print('Number of unique weights:', len(np.unique(W_tr)), 'and', len(np.unique(W_val)))

    P1 = model_baseline.predict(X_inf_std[idx_inf_1, :])

    error0_baseline = y0_error
    error1_baseline = np.mean(np.abs(Y1 - P1), 0)

    print('Error 0 baseline:', error0_baseline, 'error 1 baseline:', error1_baseline)

    baseline_model_fname = f'baseline_lr'
    baseline_model_fpath = osp.join(WINDSIDE_MODEL_DIR, baseline_model_fname)
    print('Baseline model fpath:', baseline_model_fpath)

    if not osp.isdir(baseline_model_fpath):
        os.mkdir(baseline_model_fpath)

    with open(osp.join(baseline_model_fpath, 'metainfo.json'), 'w') as f:
        json.dump({
            'input_shape': X_tr_shape,
            'output_shape': Y_tr.shape if len(Y_tr.shape) > 1 else (Y_tr.shape[0], 1),
            'x_mean': float(x_mean),
            'x_std': float(x_std),
            'y0_mean': y0_mean.tolist(),
            'y0_error': y0_error.tolist()
        }, f)

    with open(osp.join(baseline_model_fpath, 'stats.json'), 'w') as f:
        json.dump({
            'error0': error0_baseline.tolist(),
            'error1': error1_baseline.tolist(),
        }, f)

    for board in args.boards:

        model_fname = f'{board}_{args.algorithm}_{args.order}_{args.label_mode}_{args.framework}'
        model_fpath = osp.join(WINDSIDE_MODEL_DIR, model_fname)
        print('Model fpath:', model_fpath)

        if not osp.isdir(model_fpath):
            os.mkdir(model_fpath)

        with open(osp.join(model_fpath, 'metainfo.json'), 'w') as f:
            json.dump({
                'input_shape': X_tr_shape,
                'output_shape': Y_tr.shape if len(Y_tr.shape) > 1 else (Y_tr.shape[0], 1),
                'x_mean': float(x_mean),
                'x_std': float(x_std),
                'y0_mean': y0_mean.tolist(),
                'y0_error': y0_error.tolist()
            }, f)

        # main part

        dt_k, dt_b = get_dt_model_size(board=board)
        ram, flash = get_ram_flash(board=board)

        memory = ram if args.memory == 'ram' else flash

        progress = []

        trainable_data_dict = {
            'tr': [train_set_, W_tr_, W_tr],
            'val': [X_val_std_, Y_val_, W_val_, W_val],
            'inf': [X_inf_std, Y_inf, X_inf_sum]
        }

        def _train_gbdt_model(config, data=None):

            # sampled data

            for key in config.keys():
                if key.startswith('n_'):
                    config[key] = int(np.round(config[key]))

            config['random_state'] = args.seed

            params = LGBMEstimator(**config).params

            model_ = lightgbm.train(params=params, train_set=data['tr'][0])

            with BytesIO() as tmp_bytes:
                joblib.dump(model_, tmp_bytes)
                model_size = tmp_bytes.getbuffer().nbytes / 1024

            model_size_on_board = dt_k * model_size + dt_b

            print('The model size to the memory ratio:', model_size_on_board / memory)

            if model_size_on_board > memory:
                print('*********************')
                print('The model is too big!')
                print('*********************')

                error0 = error0_baseline
                error1 = error1_baseline
                #error1 = error1_too_big

            else:

                pred = model_.predict(data['inf'][0]).reshape(1, -1)
                mae = np.mean(np.abs(data['inf'][1]- pred))

                error0 = error0_baseline
                error1 = mae

            print('params:', params, ', error1:', error1)

            progress.append(f'{error0},{error1},{config["n_estimators"]},{config["num_leaves"]}\n')

            return {'error0': error0, 'error1': error1}

        def train_gbdt_model_ray(config, data=None):
            results = _train_gbdt_model(config, data)
            return results

        def train_gbdt_model_flaml(config, data=None):
            results = _train_gbdt_model(config, trainable_data_dict)
            return results

        def _train_dnn_model(config, data=None):

            # sampled data

            for key in config.keys():
                if key not in ['dropout']:
                    config[key] = int(np.round(config[key]))

            n_layers = config['n_layers']
            n_units = config['n_units']
            dropout = config['dropout']

            tf.random.set_seed(args.seed)
            tf.keras.utils.set_random_seed(args.seed)

            n_outputs = 2
            preproc_size = 0.78 + np.prod(input_shape) * 4 / 1024

            inputs = tf.keras.layers.Input(shape=input_shape)

            #inputs = tf.expand_dims(inputs, -1)

            hidden = inputs
            for i in range(n_layers):
                hidden = tf.keras.layers.Dense(units=n_units, activation='relu')(hidden)
                hidden = tf.keras.layers.Dropout(dropout)(hidden)

            #hidden = tf.keras.layers.Flatten()(hidden)

            outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(hidden)

            model_ = tf.keras.models.Model(inputs=inputs, outputs=outputs)

            model_.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), weighted_metrics=metrics)

            #model_.summary()

            layer_sizes = np.array([np.prod(input_shape)] + [n_units for _ in range(n_layers)] + [n_outputs])
            model_size = preproc_size + layer_sizes[-1] * 4 / 1024
            for i in range(len(layer_sizes) - 1):
                model_size += (layer_sizes[i] * layer_sizes[i + 1] + 3 * layer_sizes[i + 1] + layer_sizes[i]) * 4 / 1024

            print('The model size to the memory ratio: (', model_size, '/', memory, ') =', model_size / memory)

            if model_size > memory:
                print('*********************')
                print('The model is too big!')
                print('*********************')

                error_avg = error1_baseline[0]
                error_max = error1_baseline[1]
                error0 = np.mean(error0_baseline)
                error1 = np.mean(error1_baseline)

            else:

                model_.fit(
                    data['tr'][0].data, data['tr'][0].label,
                    sample_weight=pd.Series(data['tr'][1]).to_frame(),
                    validation_data=(data['val'][0], data['val'][1], data['val'][2]),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
                    ],
                    verbose=False
                )

                pred = model_.predict(data['inf'][0], verbose=False)
                mae = np.mean(np.abs(data['inf'][1] - pred), 0)

                error_avg = mae[0]
                error_max = mae[1]
                error1 = np.mean(mae)
                error0 = np.mean(error0_baseline)

            print('params:', config, 'error_avg:', error_avg, 'error_max:', error_max)

            progress.append(f'{error1},{error_avg},{error_max},{config["n_layers"]},{config["n_units"]},{config["dropout"]}\n')

            return {'error0': error0, 'error1': error1, 'error_avg': error_avg, 'error_max': error_max}

        def train_dnn_model_ray(config, data=None):
            results = _train_dnn_model(config, data)
            return results

        def train_dnn_model_flaml(config, data=None):
            results = _train_dnn_model(config, trainable_data_dict)
            return results

        def _train_cnn_model(config, data=None):

            # sampled data

            for key in config.keys():
                if key not in ['dropout']:
                    config[key] = int(np.round(config[key]))

            filters = config['filters']
            kernel = config['kernel']
            stride = config['stride']
            n_units = config['n_units']
            dropout = config['dropout']

            tf.random.set_seed(args.seed)
            tf.keras.utils.set_random_seed(args.seed)

            n_outputs = 2
            preproc_size = 0.78 + np.prod(input_shape) * 4 / 1024

            inputs = tf.keras.layers.Input(shape=input_shape)

            #inputs = tf.expand_dims(inputs, -1)

            hidden = inputs

            hidden = tf.expand_dims(inputs, -1)
            hidden = tf.keras.layers.Conv1D(filters, kernel_size=kernel, strides=stride, activation='relu')(hidden)
            hidden = tf.keras.layers.Flatten()(hidden)
            hidden = tf.keras.layers.Dropout(dropout)(hidden)
            hidden = tf.keras.layers.Dense(units=n_units, activation='relu')(hidden)
            hidden = tf.keras.layers.Dropout(dropout)(hidden)

            #hidden = tf.keras.layers.Flatten()(hidden)

            outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(hidden)

            model_ = tf.keras.models.Model(inputs=inputs, outputs=outputs)

            model_.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), weighted_metrics=metrics)

            #model_.summary()

            input_size = np.prod(input_shape)

            model_size = preproc_size + n_outputs * 4 / 1024

            # conv

            model_size += (kernel * filters + 3 * filters + input_size) * 4 / 1024
            conv_size = ((input_size - kernel) / stride + 1) * filters

            # dense

            model_size += (conv_size * n_units + 3 * n_units + conv_size) * 4 / 1024

            # output

            model_size += (n_units * n_outputs + 3 * n_outputs + n_units) * 4 / 1024

            print('The model size to the memory ratio: (', model_size, '/', memory, ') =', model_size / memory)

            if model_size > memory:
                print('*********************')
                print('The model is too big!')
                print('*********************')

                error_avg = error1_baseline[0]
                error_max = error1_baseline[1]
                error0 = np.mean(error0_baseline)
                error1 = np.mean(error1_baseline)

            else:

                model_.fit(
                    data['tr'][0].data, data['tr'][0].label,
                    sample_weight=pd.Series(data['tr'][1]).to_frame(),
                    validation_data=(data['val'][0], data['val'][1], data['val'][2]),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)
                    ],
                    verbose=False
                )

                pred = model_.predict(data['inf'][0], verbose=False)
                mae = np.mean(np.abs(data['inf'][1] - pred), 0)

                error_avg = mae[0]
                error_max = mae[1]
                error1 = np.mean(mae)
                error0 = np.mean(error0_baseline)

            print('params:', config, 'error_avg:', error_avg, 'error_max:', error_max)

            progress.append(f'{error1},{error_avg},{error_max},{config["filters"]},{config["kernel"]},{config["stride"]},{config["n_units"]},{config["dropout"]}\n')

            return {'error0': error0, 'error1': error1, 'error_avg': error_avg, 'error_max': error_max}

        def train_cnn_model_ray(config, data=None):
            results = _train_cnn_model(config, data)
            return results

        def train_cnn_model_flaml(config, data=None):
            results = _train_cnn_model(config, trainable_data_dict)
            return results

        tune = {
            'flaml': tune_flaml,
            'ray': tune_ray
        }

        if args.algorithm == 'gbdt':
            n_estimators_max, num_leaves_max = get_gbdt_model_limits(board=board)
            print('N estimators max:', n_estimators_max, 'Num leaves max:', num_leaves_max)
            config_search_space = {
                'n_estimators': tune[args.framework].uniform(1, n_estimators_max),
                'num_leaves': tune[args.framework].uniform(2, num_leaves_max),
            }

            trainable = train_gbdt_model_flaml if args.framework == 'flaml' else train_gbdt_model_ray

        elif args.algorithm == 'dnn':
            n_layers_max, n_units_max = get_dnn_model_limits(board=board)
            dropout_max = 0.5
            print('N layers max:', n_layers_max, 'n units max:', n_units_max, 'dropout max:', dropout_max)
            config_search_space = {
                'n_layers': tune[args.framework].uniform(1, n_layers_max),
                'n_units': tune[args.framework].uniform(1, n_units_max),
                'dropout': tune[args.framework].uniform(0, dropout_max)
            }

            trainable = train_dnn_model_flaml if args.framework == 'flaml' else train_dnn_model_ray

        elif args.algorithm == 'cnn':
            filters_max, kernel_max, stride_max, n_units_max = get_cnn_model_limits(board=board)
            dropout_max = 0.5
            print('Filters max:', filters_max, 'kernel max:', kernel_max, 'stride max:', stride_max, 'n units max:', n_units_max, 'dropout max:', dropout_max)
            config_search_space = {
                'filters': tune[args.framework].uniform(1, filters_max),
                'kernel': tune[args.framework].uniform(1, kernel_max),
                'stride': tune[args.framework].uniform(1, stride_max),
                'n_units': tune[args.framework].uniform(1, n_units_max),
                'dropout': tune[args.framework].uniform(0, dropout_max)
            }

            trainable = train_cnn_model_flaml if args.framework == 'flaml' else train_cnn_model_ray

        if args.framework == 'flaml':

            tuner = tune_flaml.run(
                trainable, metric="error1", mode="min", config=config_search_space, search_alg=BlendSearch(), time_budget_s=None, num_samples=args.trials,
            )

            print('Best config:', tuner.best_config)
            print('Best results:', tuner.best_result)

            best_config = tuner.best_config

            with open(osp.join(model_fpath, 'progress.csv'), 'w') as f:
                f.writelines(progress)

        elif args.framework == 'ray':

            tuner = tune_ray.Tuner(
                tune_ray.with_parameters(trainable, data=trainable_data_dict),
                param_space=config_search_space,
                tune_config=tune_ray.TuneConfig(
                    search_alg=BayesOptSearch(),
                    metric="error1", mode="min",
                    num_samples=args.trials,
                    max_concurrent_trials=1
                ),
                run_config=air.config.RunConfig(verbose=args.verbose)
            )
            tuner.fit()

            best_config = tuner.get_results().get_best_result().config
            print('Best results:', tuner.get_results().get_best_result().metrics)

            df = tuner.get_results().get_dataframe()
            errors1 = df['error1'].values
            errors_avg = df['error_avg'].values
            errors_max = df['error_max'].values

            if args.algorithm == 'gbdt':
                hp_keys = ['config/n_estimators', 'config/num_leaves']
            elif args.algorithm == 'dnn':
                hp_keys = ['config/n_layers', 'config/n_units', 'config/dropout']
            elif args.algorithm == 'cnn':
                hp_keys = ['config/filters', 'config/kernel', 'config/stride', 'config/n_units', 'config/dropout']

            hps = df[hp_keys].values

            progress = []
            for i, (error1, error_avg, error_max) in enumerate(zip(errors1, errors_avg, errors_max)):
                hp_line = ','.join([str(hp) for hp in hps[i, :]])
                progress.append(f'{error1},{error_avg},{error_max},{hp_line}\n')

            with open(osp.join(model_fpath, 'progress.csv'), 'w') as f:
                f.writelines(progress)

        else:
            raise NotImplemented

        if args.algorithm == 'gbdt':

            params = LGBMEstimator(**best_config).params
            print('Params:', params)
            model = lightgbm.sklearn.LGBMRegressor(**params)
            model.fit(X_tr_std, Y_tr)
            joblib.dump(model, osp.join(model_fpath, 'model.joblib'))

        elif args.algorithm == 'dnn':

            params = best_config
            inputs = tf.keras.layers.Input(shape=input_shape)
            hidden = inputs
            for i in range(params['n_layers']):
                hidden = tf.keras.layers.Dense(units=params['n_units'], activation='relu')(hidden)
                hidden = tf.keras.layers.Dropout(params['dropout'])(hidden)
            outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(hidden)
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), weighted_metrics=metrics)
            model.fit(
                X_tr_std, Y_tr,
                sample_weight=pd.Series(W_tr).to_frame(),
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

        elif args.algorithm == 'cnn':

            params = best_config
            inputs = tf.keras.layers.Input(shape=input_shape)

            hidden = tf.expand_dims(inputs, -1)
            hidden = tf.keras.layers.Conv1D(params['filters'], kernel_size=params['kernel'], strides=params['stride'], activation='relu')(hidden)
            hidden = tf.keras.layers.Flatten()(hidden)
            hidden = tf.keras.layers.Dropout(params['dropout'])(hidden)
            hidden = tf.keras.layers.Dense(units=params['n_units'], activation='relu')(hidden)
            hidden = tf.keras.layers.Dropout(params['dropout'])(hidden)

            outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(hidden)
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), weighted_metrics=metrics)
            model.fit(
                X_tr_std, Y_tr,
                sample_weight=pd.Series(W_tr).to_frame(),
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

        print(f'Saved to {model_fpath}')

        if X_inf.shape[0] > 0:

            X_inf_sum = np.array([np.sum(x) for x in X_inf])
            idx_inf_0 = np.where(X_inf_sum == 0)[0]
            idx_inf_1 = np.where(X_inf_sum > 0)[0]
            print(len(idx_inf_0), len(idx_inf_1))
            Y0 = Y_inf[idx_inf_0, :]
            Y1 = Y_inf[idx_inf_1, :]

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
                }, f)

            pd.DataFrame(error1_arr_sorted).to_csv(osp.join(model_fpath, 'error.csv'), header=None, index=None)