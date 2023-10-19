import numpy as np
import argparse as arp
import os.path as osp
import tensorflow as tf

from utils.preprocess_data import load_dataset, split_data
from config import *

if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Test supervised methods.')
    parser.add_argument('-f', '--feature_extractors', help='Feature extractors', nargs='+', default=['flat'])
    parser.add_argument('-d', '--dataset', help='Dataset name', default='pump_raw')
    args = parser.parse_args()

    dataset = args.dataset
    if dataset.startswith('fan'):
        labels = {0: ['normal', 'on_off'], 1: ['stick', 'tape', 'shake']}
    elif dataset.startswith('bearing'):
        labels = {0: ['normal'], 1: ['crack', 'sand']}
    elif dataset.startswith('pump'):
        labels = {0: ['normal'], 1: ['fast']}

    data_fpath = osp.join(ARE_DATA_DIR, dataset)
    target_dataset = load_dataset(data_fpath, series_len=1, series_step=1, labels=labels, feature_extractors=args.feature_extractors)
    data = split_data(target_dataset, train_on_anomalies=True, validate_on_anomalies=True, shuffle_features=False)

    inp_shape = data['tr'][0].shape[1:]

    print(data['tr'][0].shape)

    model_fpath = osp.join('model_autokeras', args.dataset, *[str(fe) for fe in args.feature_extractors])

    try:
        model = tf.keras.models.load_model(model_fpath)
        model.summary()

    except Exception as e:
        print(e)

        inputs = tf.keras.layers.Input(shape=inp_shape)
        hidden = tf.keras.layers.Normalization()(inputs)
        if len(inp_shape) == 2:
            hidden = tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=1, activation='relu')(hidden)
            hidden = tf.keras.layers.Dropout(0.5)(hidden)
            hidden = tf.keras.layers.Flatten()(hidden)
        # hidden = tf.keras.layers.Dense(units=32, activation='relu')(hidden)
        hidden = tf.keras.layers.Dense(units=32)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        # hidden = tf.keras.layers.Dropout(0.0)(hidden)
        # hidden = tf.keras.layers.Dense(units=512, activation='relu')(hidden)
        hidden = tf.keras.layers.Dense(units=512)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        # hidden = tf.keras.layers.Dropout(0.0)(hidden)
        # hidden = tf.keras.layers.Dense(units=32, activation='relu')(hidden)
        hidden = tf.keras.layers.Dense(units=32)(hidden)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden)
        # hidden = tf.keras.layers.Dropout(0.0)(hidden)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(hidden)
        # outputs = tf.keras.layers.Dense(1)(hidden)

        model = tf.keras.models.Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics='binary_accuracy')
        model.summary()

        model.fit(
            *data['tr'],
            validation_data=data['val'],
            epochs=10000,
            batch_size=2048,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
            ]
        )

    h = model.evaluate(*data['inf'])
    print(h)
