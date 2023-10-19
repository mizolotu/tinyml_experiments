import json, uuid, multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse as arp

from config import *
from windside_extract_data import create_dataset, split_dataset


def parse_layers(inputs, layers, dropout_rate=0.5):
    last_shape = inputs.shape
    hidden = inputs
    if layers is not None:
        for i, layer in enumerate(layers):
            #print(i, layer)
            spl = layer.split('_')
            layer_type = spl[0]
            if layer_type == 'dense':
                units = int(spl[1])
                if len(last_shape) > 2:
                    hidden = tf.keras.layers.Flatten()(hidden)
                hidden = tf.keras.layers.Dense(units, activation='relu')(hidden)
            elif layer_type == 'lstm':
                assert len(last_shape) == 3
                units = int(spl[1])
                if i < len(layers) - 1 and (layers[i + 1].startswith('lstm') or layers[i + 1].startswith('conv')):
                    return_seq = True
                else:
                    return_seq = False
                hidden = tf.keras.layers.LSTM(units, activation='relu', return_sequences=return_seq)(hidden)
            elif layer_type == 'conv':
                assert len(last_shape) == 3
                filters = int(spl[1])
                kernel = int(spl[2]) # inputs.shape[1] // 8
                stride = int(spl[3]) #kernel // 2
                #print(kernel, stride)
                hidden = tf.keras.layers.Conv1D(filters, kernel_size=kernel, strides=stride, activation='relu')(hidden)
            hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
    return hidden

def arch_test(seed, arch, order, label_mode, weights, x_input_shape, t_input_shape,
        output_dim, loss, epochs, patience, lr, batch_size, val_split, dropout,
        x_mean, x_std, y0_mean, y0_error,
        X_tr, T_tr, Y_tr, W_tr, X_inf, T_inf, Y_inf):

    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    print('Testing model architecture:', arch)

    if type(arch) is dict:
        x_embed_layers = arch['x'] if 'x' in arch.keys() else None
        t_embed_layers = arch['t'] if 't' in arch.keys() else None
        common_layers = arch['c'] if 'c' in arch.keys() else None

    metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

    x_inputs = tf.keras.layers.Input(shape=x_input_shape)
    x_hidden = (x_inputs - x_mean) / (x_std + 1e-10)
    x_hidden = parse_layers(x_hidden, x_embed_layers, dropout_rate=dropout)

    if t_embed_layers is not None:
        t_inputs = tf.keras.layers.Input(shape=t_input_shape)
        t_hidden = parse_layers(t_inputs, t_embed_layers, dropout_rate=dropout)
    else:
        t_inputs = None
        t_hidden = []

    if x_embed_layers is not None and t_embed_layers is not None:
        if len(x_hidden.shape) > 2:
            x_hidden = tf.keras.layers.Flatten()(x_hidden)
        if len(t_hidden.shape) > 2:
            t_hidden = tf.keras.layers.Flatten()(t_hidden)
        hidden = tf.concat([x_hidden, t_hidden], axis=1)
    elif x_embed_layers is not None:
        hidden = x_hidden
    elif t_embed_layers is not None:
        hidden = t_hidden
    else:
        mode = 'baseline'
        hidden = None

    model_fname = f'tf_{uuid.uuid3(uuid.NAMESPACE_DNS, json.dumps(arch))}_{order}_{label_mode}'
    if weights is not None:
        model_fname = f'{model_fname}_{weights}'
    # f'dense_{"nozero" if filter_zeros else "all"}_{"voltage" if include_voltage else "current"}_{window_size}_{series_step}'
    model_fpath = osp.join(WINDSIDE_MODEL_DIR, model_fname)
    print('Model fpath:', model_fpath)

    if hidden is not None:
        hidden = parse_layers(hidden, common_layers, dropout_rate=dropout)

        outputs = tf.keras.layers.Dense(units=output_dim, activation='linear')(hidden)

        if t_inputs is not None:
            inputs = [x_inputs, t_inputs]
        else:
            inputs = x_inputs

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        model_compile_kwargs = {}
        if weights is None:
            model_compile_kwargs['metrics'] = metrics
        else:
            model_compile_kwargs['weighted_metrics'] = metrics

        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr), **model_compile_kwargs)

        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        summary = "\n".join(model_summary)

        print(summary)

        model_fit_kwargs = {
            'validation_split': val_split,
            'epochs': epochs,
            'batch_size': batch_size,
            'shuffle': True,
            'callbacks': [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)],
            'verbose': True
        }
        if weights is not None:
            model_fit_kwargs['sample_weight'] = pd.Series(W_tr).to_frame()

        model.fit([X_tr, T_tr] if t_inputs is not None else X_tr, Y_tr, **model_fit_kwargs)

        # quick test
        #print(f'Quick test result: {model.predict(X_tr[[0]])}')

        model.save(model_fpath)
        print(f'Saved to {model_fpath}')

        with open(osp.join(model_fpath, 'summary.txt'), 'w') as fp:
            fp.write(summary)

    if not osp.isdir(model_fpath):
        os.mkdir(model_fpath)

    with open(osp.join(model_fpath, 'metainfo.json'), 'w') as f:
        json.dump({
            'input_shape': X_tr.shape,
            'output_shape': Y_tr.shape if len(Y_tr.shape) > 1 else (Y_tr.shape[0], 1),
            'x_mean': float(x_mean),
            'x_std': float(x_std),
            'y0_mean': y0_mean.tolist(),
            'y0_error': y0_error.tolist()
        }, f)

    if X_inf.shape[0] > 0:

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        X_inf_sum = np.array([np.sum(x) for x in X_inf[:, :, 0]])
        idx_inf_1 = np.where(X_inf_sum > 0)[0]
        print('Number of positive current samples in the inference subset =', len(idx_inf_1))
        Y1 = Y_inf[idx_inf_1]

        P1 = model.predict(
            [X_inf[idx_inf_1, :], T_inf[idx_inf_1, :]] if t_inputs is not None else X_inf[idx_inf_1, :],
            batch_size=batch_size
        )

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
                'model': json.dumps(arch)
            }, f)

        pd.DataFrame(error1_arr_sorted).to_csv(osp.join(model_fpath, 'error.csv'), header=None, index=None)

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Train wind speed tf model.')
    parser.add_argument('-s', '--seed', help='Seed', type=int, default=42)
    parser.add_argument('-o', '--order', help='Order', default='date', choices=['random', 'date', 'windspeed'])
    parser.add_argument('-l', '--label_mode', help='Label mode', default='avg+max', choices=['avg', 'max', 'avg+max'])
    parser.add_argument('-w', '--weights', help='Weight type', default=None, choices=[None, 'discrete', 'continuous'])
    parser.add_argument('-c', '--chunk_size', help='Chunk size', type=int, default=None)
    parser.add_argument('-a', '--architectures', help='Model architecture', nargs='+', default=[
        #'{}',

        #'{"x": ["dense_64", "dense_64"], "c": []}',
        #'{"x": ["conv_24_20_10", "dense_64"], "c": []}',
        #'{"x": ["lstm_36", "dense_64"], "c": []}',

        '{"x": ["dense_246"], "c": []}',

        #'{"x": ["dense_32"], "t": ["dense_32"], "c": ["dense_64"]}',
        #'{"x": ["dense_64"], "t": [], "c": ["dense_64"]}',
    ])
    parser.add_argument('-g', '--gpu', help='GPU', default='0')
    args = parser.parse_args()

    include_voltage = False
    seed = args.seed

    series_step = 1
    window_size = 60

    inf_split = 0.2

    loss = 'mean_squared_error'
    # loss = 'mean_absolute_error'
    epochs = 1000
    patience = 10
    lr = 1e-3
    batch_size = 4096
    val_split = 0.2
    dropout = 0.5

    data_fpath = osp.join(DATA_DIR, 'windside', 'features.csv')
    dataset_fpath_pattern = osp.join(DATA_DIR, 'windside', '{0}.{1}')

    feature_cols = ['Current']
    if include_voltage:
        feature_cols.append('Voltage')

    np.random.seed(seed)

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

    output_dim = Y.shape[1]

    print(np.min(Y, 0), np.max(Y, 0))

    X_tr, T_tr, Y_tr, D_tr, W_tr, X_val, T_val, Y_val, D_val, W_val, X_inf, T_inf, Y_inf, D_inf, W_inf = split_dataset(
        (X, T, Y, D, W),
        inf_split=inf_split,
        order=args.order,
        shuffle_train=False
    )

    print('Number of unique weights:', len(np.unique(W_tr)), 'and', len(np.unique(W_val)))

    print('Shapes: ', X_tr.shape, T_tr.shape, Y_tr.shape, X_inf.shape, T_inf.shape, Y_inf.shape)

    print('The training set dates:', D_tr[0], D_tr[-1])
    print('The validation set dates:', D_val[0], D_val[-1])
    if len(D_inf) > 0:
        print('The inference set dates:', D_inf[0], D_inf[-1])

    print('The training set wind speeds:', Y_tr[0], Y_tr[-1])
    print('The validation set wind speeds:', Y_val[0], Y_val[-1])
    if len(Y_inf) > 0:
        print('The inference set wind speeds:', Y_inf[0], Y_inf[-1])

    x_input_shape = X_tr.shape[1:]
    t_input_shape = T_tr.shape[1:]

    if args.architectures is None or len(args.architectures) == 0:
        architectures = [None]
    else:
        architectures = args.architectures

    x_mean = np.mean(X_tr)
    x_std = np.std(X_tr)

    if len(architectures) == 1:

        architecture = architectures[0]

        arch = None
        exec(f'arch = {architecture}')

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        arch_test(
            seed,
            arch, args.order, args.label_mode, args.weights, x_input_shape, t_input_shape,
            output_dim, loss, epochs, patience, lr, batch_size, val_split, dropout,
            x_mean, x_std, y0_mean, y0_error,
            X_tr, T_tr, Y_tr, W_tr, X_inf, T_inf, Y_inf
        )

    else:

        for architecture in args.architectures:

            arch = None
            exec(f'arch = {architecture}')

            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

            p = multiprocessing.Process(target=arch_test, args=(
                seed,
                arch, args.order, args.label_mode, args.weights, x_input_shape, t_input_shape,
                output_dim, loss, epochs, patience, lr, batch_size, val_split, dropout,
                x_mean, x_std, y0_mean, y0_error,
                X_tr, T_tr, Y_tr, W_tr, X_inf, T_inf, Y_inf
            ))

            p.start()
            p.join()