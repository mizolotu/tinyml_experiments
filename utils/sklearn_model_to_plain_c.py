import numpy as np

import m2cgen


def generate_model_lines(model, metadata, include_metadata=['x_mean', 'x_std', 'y0_mean'], board='arduino'):

    if board == 'arduino':
        include_avr_pgmspace = '#include <avr/pgmspace.h>'
        progmem = 'PROGMEM'
    elif board == 'lora':
        include_avr_pgmspace = '#define pgm_read_float_near(addr) (*(const float *)(addr))'
        progmem = ''
    else:
        raise NotImplemented

    assert 'x_mean' in metadata.keys()
    x_mean = metadata['x_mean']
    assert 'x_std' in metadata.keys()
    x_std = metadata['x_std']

    include_lines = [
        include_avr_pgmspace,
        ''
    ]

    # input size

    if 'input_shape' in metadata.keys():
        input_shape = metadata['input_shape'][1:]
    elif 'input_shape' in dir(model):
        input_shape = np.array(model.input_shape[1:])
    elif 'n_features_in_' in dir(model):
        input_shape = [model.n_features_in_]
    else:
        raise NotImplemented

    if len(input_shape) == 1:
        size_lines = [f'\nconst int INPUT_SIZE = {input_shape[0]};\n']
    elif len(input_shape) == 2 and input_shape[1] == 1:
        size_lines = [f'\nconst int INPUT_SIZE = {input_shape[0]};\n']

    # metadata

    data_lines = []
    read_lines = []

    for key in metadata.keys():

        if key in include_metadata:
            data_lines.append(f'const float {key}_data[] {progmem} = {{')
            value = metadata[key]
            if type(value) == list:
                x_str = ','.join([f'{item:.8f}f' for item in value])
            else:
                x_str = f'{value:.8f}f'
            data_lines.append(x_str)
            data_lines.append('};')

    # meta read

    for key in metadata.keys():
        if key in include_metadata:
            read_lines.extend([
                f'inline float {key}(int i) {{',
                f'return pgm_read_float_near({key}_data + i);',
                '}\n'
            ])

    # score function

    score_lines = [m2cgen.export_to_c(model).replace('double', 'float')]

    # predict function

    predict_lines = [
        'void predict(float* x, float* y) {',
        '',
        '  float x_sum = 0.0;',
        '',
        '  for (int i = 0; i < INPUT_SIZE; i++) {',
        f'    x_sum += x[i];',
        '  }',
        '',
        '  if (x_sum == 0) {',
        '',
        '    y[0] = y0_mean(0);',
        '',
        '  } else {',
        '',
        '    float x_nrm[INPUT_SIZE];',
        '    for (int i = 0; i < INPUT_SIZE; i++) {',
        f'      x_nrm[i] = (x[i] - {"x_mean(i)" if type(x_mean) == list else "x_mean(0)"}) / {"x_std(i)" if type(x_std) == list else "x_std(0)"};',
        '    }',
        '    y[0] = score(x_nrm);',
        '  }',
        '}'
    ]

    lines = include_lines + size_lines + data_lines + read_lines + score_lines + predict_lines

    return lines



