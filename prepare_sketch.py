import json, joblib

import argparse as arp
import tensorflow as tf

from utils.tf_model_to_plain_c import generate_model_lines as generate_tf_lines
from utils.sklearn_model_to_plain_c import generate_model_lines as generate_skl_lines
from utils.generate_sketch import generate_arduino_lines
from config import *


if __name__ == '__main__':

    parser = arp.ArgumentParser(description='Prepare arduino/lora sketch.')
    parser.add_argument('-t', '--task', help='Task name', default='windside')
    parser.add_argument('-m', '--model', help='Model name', default='lora_dnn_auto_ray_weighted_gpuserv')
    #parser.add_argument('-m', '--model', help='Model name', default='lora_dnn_auto_ray')
    #parser.add_argument('-m', '--model', help='Model name', default='lgbmr_auto_filter-zeros_mean')
    parser.add_argument('-b', '--board', help='Board', default='lora')
    args = parser.parse_args()

    if args.task == 'windside':
        task_model_dir = WINDSIDE_MODEL_DIR
    else:
        raise NotImplemented

    board_name = args.board

    model_name = args.model
    model_fpath = osp.join(task_model_dir, model_name)
    meta_fpath = osp.join(model_fpath, 'metainfo.json')

    postfix = '_'.join(args.model.split('_')[:2])
    sketch_name = f'{args.task}_{postfix}'

    sketch_ino_fpath = osp.join(SKETCH_DIR, board_name, sketch_name, f'{sketch_name}.ino')
    model_h_fpath = osp.join(SKETCH_DIR, board_name, sketch_name, 'model.h')

    if not osp.isdir(osp.join(SKETCH_DIR, board_name)):
        os.mkdir(osp.join(SKETCH_DIR, board_name))

    if not osp.isdir(osp.join(SKETCH_DIR, board_name, sketch_name)):
        os.mkdir(osp.join(SKETCH_DIR, board_name, sketch_name))

    model, model_type = None, None

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

        sketch_lines = generate_arduino_lines(model, metadata)

        with open(sketch_ino_fpath, 'w') as f:
            f.writelines([f'{line}\n' for line in sketch_lines])

        model_lines = generate_tf_lines(model, metadata, board=args.board) if model_type == 'tf' else generate_skl_lines(model, metadata, board=args.board)

        with open(model_h_fpath, 'w') as f:
            f.writelines([f'{line}\n' for line in model_lines])

    else:
        print('Model should be either created in Tensorflow or in one of the following packages: Sklearn, Lightning, XGBoost or LightGBM!')