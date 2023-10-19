import tensorflow as tf
import numpy as np


def relu_lines():
    lines = [
        '',
        'inline float relu(float x) {',
        '  return fmaxf(0.0f, x);',
        '}'
    ]
    return lines

def linear_lines():
    lines = [
        '',
        'inline float linear(float x) {',
        '  return x;',
        '}'
    ]
    return lines

def sigmoid_lines():
    lines = [
        '',
        'inline float sigmoid(float x) {',
        '  return 1.0 / (1.0 + exp(-x));',
        '}'
    ]
    return lines

def generate_model_lines(model, metadata, include_metadata=['x_mean', 'x_std', 'y0_mean', 'y_min', 'y_max'], board='lora'):

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
    assert 'y_min' in metadata.keys()
    y_min = metadata['y_min']
    assert 'y_max' in metadata.keys()
    y_max = metadata['y_max']

    include_lines = [
        include_avr_pgmspace,
        ''
    ]

    # add input size

    input_shape = model.input_shape[1:]
    if len(input_shape) == 1:
        size_lines  = [f'\nconst int INPUT_SIZE = {input_shape[0]};\n']
    elif len(input_shape) == 2 and input_shape[1] == 1:
        size_lines = [f'\nconst int INPUT_SIZE = {input_shape[0]};\n']

    # add metadata

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
            data_lines.append(f'{x_str}')
            data_lines.append('};')

    # add activation and layer data

    layers = []
    activation_names = []
    activation_lines = []
    for layer in model.layers:

        if 'activation' in dir(layer):
            activation_name = layer.activation.__name__
            if activation_name not in activation_names:
                #print(activation_name)
                activation_names.append(activation_name)
                activation_lines_fun = f'{activation_name}_lines'
                assert activation_lines_fun in globals()
                activation_lines.extend(globals()[activation_lines_fun]())

        if 'weights' in dir(layer) and 'bias' in dir(layer):
            layers.append(layer)

    # add meta read

    for key in metadata.keys():
        if key in include_metadata:
            read_lines.extend([
                '',
                f'inline float {key}(int i) {{',
                f'  return pgm_read_float_near({key}_data + i);',
                '}'
            ])

    # parse model

    predict_lines = [
        '',
        'void predict(float* x, float* y) {',
        ''
    ]

    predict_lines.extend([
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
    ])

    last_layer_name = 'INPUT'

    for layer_i, layer in enumerate(layers):

        if isinstance(layer, tf.keras.layers.Dense):

            #print(layer.name, layer.activation.__name__)
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]

            #print('w:', w.shape)

            predict_lines.extend([
                f'    float h{layer_i}[DENSE{layer_i}_SIZE];\n',
                f'    for (int i = 0; i < DENSE{layer_i}_SIZE; ++i) {{',
                f'      h{layer_i}[i] = 0.0;',
                f'      for (int j = 0; j < {last_layer_name}_SIZE; ++j) {{'
            ])
            if layer_i == 0:
                line = f'        h{layer_i}[i] += (x[j] - {"x_mean(j)" if type(x_mean) == list else "x_mean(0)"}) / {"x_std(j)" if type(x_std) == list else "x_std(0)"} * W{layer_i}(j, i);'
            else:
                line = f'        h{layer_i}[i] += h{layer_i - 1}[j] * W{layer_i}(j, i);'
            predict_lines.append(line)
            predict_lines.append('      }')
            if layer_i < len(layers) - 1:
                predict_lines.append(f'      h{layer_i}[i] = {layer.activation.__name__}(h{layer_i}[i] + b{layer_i}(i));')
            else:
                if layer.activation.__name__ == 'linear':
                    predict_lines.append(f'      y[i] = {layer.activation.__name__}(h{layer_i}[i] + b{layer_i}(i));')
                else:
                    predict_lines.append(f'      y[i] = {layer.activation.__name__}(h{layer_i}[i] + b{layer_i}(i)) * ({"(y_max(j)" if type(y_max) == list else "y_max(0)"} - {"(y_min(j)" if type(y_min) == list else "y_min(0)"}) + {"(y_min(j)" if type(y_min) == list else "y_min(0)"};')
            predict_lines.append('    }\n')

            size_lines.append(f'const int DENSE{layer_i}_SIZE = {layer.units};\n')
            read_lines.extend([
                '',
                f'inline float W{layer_i}(int i, int j) {{',
                f'  return pgm_read_float_near(W{layer_i}_data + i * DENSE{layer_i}_SIZE + j);',
                '}',
                '',
                f'inline float b{layer_i}(int i) {{',
                f'  return pgm_read_float_near(b{layer_i}_data + i);',
                '}'
            ])

            data_lines.append(f'const float W{layer_i}_data[] {progmem} = {{')
            w = np.array(w)
            w = w.reshape(1, -1).flatten()
            w_str = ','.join([f'{item:.8f}f' for item in w])
            data_lines.append(w_str)
            data_lines.append('};')

            #print('b:', b.shape)
            data_lines.append(f'const float b{layer_i}_data[] {progmem} = {{')
            b = np.array(b)
            b_str = ','.join([f'{item:.8f}f' for item in b])
            data_lines.append(b_str)
            data_lines.append('};')

            last_layer_name = f'DENSE{layer_i}'

        elif isinstance(layer, tf.keras.layers.Conv1D):

            # TO DO:

            #print(layer)

            w = layer.get_weights()[0]
            b = layer.get_weights()[1]

            #print('w:', w.shape)
            data_lines.append(f'const float W{layer_i}_data[] {progmem} = {{')
            w = np.array(w)
            w = w.reshape(1, - 1).flatten()
            w_str = ','.join([f'{item:.8f}f' for item in w])
            data_lines.append(w_str)
            data_lines.append('};')

            #print('b:', b.shape)
            data_lines.append(f'const float b{layer_i}_data[] {progmem} = {{')
            b = np.array(b)
            b_str = ','.join([f'{item:.8f}f' for item in b])
            data_lines.append(b_str)
            data_lines.append('};')

            last_layer_name = f'CONV1D{layer_i}'

    predict_lines.extend([
        '  }',
        '}'
    ])

    lines = include_lines + size_lines + data_lines + read_lines + activation_lines + predict_lines

    return lines