import numpy as np

def generate_arduino_lines(model, metadata):

    if 'input_shape' in metadata.keys():
        input_shape = metadata['input_shape'][1:]
    elif 'input_shape' in dir(model):
        input_shape = np.array(model.input_shape[1:])
    elif 'n_features_in_' in dir(model):
        input_shape = [model.n_features_in_]
    else:
        raise NotImplemented

    if 'output_shape' in metadata.keys():
        output_shape = metadata['output_shape'][1:]
    elif 'output_shape' in dir(model):
        output_shape = np.array(model.output_shape[1:])
    else:
        output_shape = [1]

    if len(input_shape) == 1 or len(input_shape) > 1 and np.all(input_shape[1:] == 1):
        input_dim = input_shape[0]
    else:
        raise NotImplemented

    if len(output_shape) == 1:
        output_dim = output_shape[0]
    else:
        raise NotImplemented

    lines = [
        '#include "model.h"',
        '',
        '#define SLEEP_INTERVAL 1000',
        '#define PREDICT_AFTER     1',
        ''
    ]

    if 'x_test' in metadata.keys():
        x_test = metadata['x_test']
    else:
        assert 'x_mean' in metadata.keys()
        x_mean = metadata['x_mean']
        assert 'x_std' in metadata.keys()
        x_std = metadata['x_std']
        x_test = x_mean + np.random.randn(input_dim) * x_std

    x_test_str = ','.join([f'{item:.8f}f' for item in x_test])
    lines.append(f'float x[INPUT_SIZE], y[{output_dim}];')
    lines.append('unsigned short step;')
    lines.append('')
    lines.append(f'float x_test[INPUT_SIZE] = {{{x_test_str}}};')
    lines.append('')

    lines.extend([
        '',
        'void setup() {',
        '',
        '  Serial.begin(115200);',
        '  while (!Serial);',
        '',
        '  for (int i = 0; i < INPUT_SIZE; i++) {',
        '    x[i] = 0.0;',
        '  }',
        '',
        '  step = 0;',
        '',
        '}',
        '',
        'void loop() {',
        '',
        '  // move previous inputs one position to the left, append a new value to the end of the input array',
        '',
        '  for (int i = 0; i < INPUT_SIZE - 1; i++) {',
        '    x[i] = x[i + 1];',
        '  }',
        '  x[INPUT_SIZE - 1] = x_test[step]; // <--- here should go a new value (instead of x_test[step])',
        '',
        '  // check whether it is time to predict or not',
        '',
        '  if ((step + 1) % PREDICT_AFTER == 0) {',
        '',
        '    predict(x, y);',
        '',
        '    Serial.print(*(y), 8);',
        '    Serial.println("");',
        '',
        '  }',
        '',
        '  step = (step + 1) % INPUT_SIZE;',
        '',
        '  delay(SLEEP_INTERVAL);',
        '',
        '}'
    ])

    return lines