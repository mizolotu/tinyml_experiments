def get_ram_flash(board='sense'):
    if board == 'sense':
        r = 256
        f = 1024
    elif board == 'iot':
        r = 32
        f = 256
    elif board == 'every':
        r = 6
        f = 48
    elif board == 'uno':
        r = 2
        f = 32
    elif board == 'lora':
        r = 64
        f = 256
    else:
        raise NotImplemented
    return r, f

def get_dt_model_size(board='sense'):
    if board == 'sense':
        k = 0.34
        b = 84
    elif board == 'iot':
        k = 0.35
        b = 21
    elif board == 'every':
        k = 0.4
        b = 4.7
    elif board == 'uno':
        k = 0.4
        b = 3.6
    elif board == 'lora':
        k = 0.37
        b = 107
    else:
        raise NotImplemented
    return k, b

def get_gbdt_model_limits(board='sense'):
    if board == 'sense':
        n_estimators = 256
        num_leaves = 128
    elif board == 'iot':
        n_estimators = 128
        num_leaves = 64
    elif board == 'every':
        n_estimators = 64
        num_leaves = 32
    elif board == 'uno':
        n_estimators = 32
        num_leaves = 16
    elif board == 'lora':
        n_estimators = 128
        num_leaves = 64
    else:
        raise NotImplemented
    return n_estimators, num_leaves

def get_dnn_model_limits(board='sense'):
    if board == 'sense':
        n_layers = 3
        n_units = 256
    elif board == 'iot':
        n_layers = 3
        n_units = 96
    elif board == 'every':
        n_layers = 3
        n_units = 64
    elif board == 'uno':
        n_layers = 3
        n_units = 32
    elif board == 'lora':
        n_layers = 3
        n_units = 128
    else:
        raise NotImplemented
    return n_layers, n_units

def get_cnn_model_limits(board='sense'):
    if board == 'sense':
        filters = 32
        kernel = 20
        stride = 20
        n_units = 128
    elif board == 'iot':
        filters = 16
        kernel = 20
        stride = 20
        n_units = 32
    elif board == 'every':
        filters = 8
        kernel = 20
        stride = 20
        n_units = 32
    elif board == 'uno':
        filters = 8
        kernel = 20
        stride = 20
        n_units = 16
    elif board == 'lora':
        filters = 32
        kernel = 20
        stride = 20
        n_units = 64
    else:
        raise NotImplemented
    return filters, kernel, stride, n_units