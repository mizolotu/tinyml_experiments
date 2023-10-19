import os
import os.path as osp

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = 'data'
ARE_DATA_DIR = osp.join(DATA_DIR, 'are')
BENCHMARK_DATA_DIR = osp.join(ARE_DATA_DIR, 'benchmarks')
FAN_DATA_DIR = osp.join(ARE_DATA_DIR, 'fan')
BEARING_DATA_DIR = osp.join(ARE_DATA_DIR, 'bearing_fft_std')
WINDSIDE_DATA_DIR = osp.join(DATA_DIR, 'windside')

MODEL_DIR = 'models'
WINDSIDE_MODEL_DIR = osp.join(MODEL_DIR, 'windside')
ARE_MODEL_DIR = osp.join(MODEL_DIR, 'are')

FIG_DIR = 'figures'
WINDSIDE_FIG_DIR = osp.join(FIG_DIR, 'windside')
ARE_FIG_DIR = osp.join(FIG_DIR, 'are')

RESULT_DIR = 'results'

SKETCH_DIR = 'sketches'
UNO_SKETCH_DIR = osp.join(SKETCH_DIR, 'uno')
EVERY_SKETCH_DIR = osp.join(SKETCH_DIR, 'every')
IOT_SKETCH_DIR = osp.join(SKETCH_DIR, 'iot')
LORA_SKETCH_DIR = osp.join(SKETCH_DIR, 'lora')
NANO_SKETCH_DIR = osp.join(SKETCH_DIR, 'nano')

SKETCH_TEST_DIR = 'sketch_tests'