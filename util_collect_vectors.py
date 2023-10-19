import os

import serial, pandas, argparse
import numpy as np
import os.path as osp

from datetime import datetime

def receive_vector(ser, start_marker=60, end_marker=62):

    msg = ''
    x = 'z'
    while ord(x) != start_marker:
        x = ser.read()

    while ord(x) != end_marker:
        if ord(x) != start_marker:
            msg = f'{msg}{x.decode("utf-8")}'
        x = ser.read()

    result_dict = {}
    if '=' in msg:
        try:
            spl = msg.split(';')
            for part in spl:
                part_spl = part.split('=')
                key = part_spl[0]
                value = [float(item) for item in part_spl[1].split(',')]
                result_dict[key] = value
        except Exception as e:
            print(e)
    else:
        if 'Important: ' in msg:
            result_dict['m'] = f"{datetime.now().strftime('%d.%m.%y %H:%M:%S.%f')}: {msg.split('Important: ')[1].capitalize()}"
            print(result_dict['m'])

    return result_dict, msg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse args')
    parser.add_argument('-p', '--port', help='Serial port', default='/dev/ttyACM3')
    parser.add_argument('-r', '--rate', help='Baud rate', default=115200, type=int)
    parser.add_argument('-s', '--start', help='Start marker', default=60, type=int)
    parser.add_argument('-e', '--end', help='End marker', default=62, type=int)
    parser.add_argument('-x', '--xdata', help='Data sample identifier', default='i')
    parser.add_argument('-b', '--baseline', help='Number of vectors to record as a baseline', default=0, type=int)
    parser.add_argument('-n', '--nvectors', help='Number of vectors to record', default=10000, type=int)
    parser.add_argument('-f', '--fnames', help='File names', nargs='+', default=['1', '2', '3', '4', '5'])
    parser.add_argument('-l', '--label', help='File path', default='normal')
    parser.add_argument('-d', '--directory', help='Directory to store the dataset', default='data/are/bearing_fft_std_rnd')
    args = parser.parse_args()


    ser = serial.Serial(args.port, args.rate)

    for fname in args.fnames:

        input(f'Press Enter to record the next file, i.e. {fname}...')

        # record the baseline

        if args.baseline is not None and args.baseline > 0:
            data = []
            n = 0
            while n < args.baseline:
                x_dict, msg = receive_vector(ser, args.start, args.end)
                if x is not None:
                    print(n, x)
                    data.append(x)
                    n += 1
                else:
                    print(msg)
            B = np.array(data)
            b_mean = np.mean(B, 0)

            print('Baseline data samples have been recorded!')
            input('Press Enter to continue...')

        else:
            b_mean = None

        # record the data

        data = []

        n = 0
        while n < args.nvectors:
            x_dict, msg = receive_vector(ser, args.start, args.end)
            if args.xdata in x_dict.keys():
                x = x_dict[args.xdata]
                print(n, x)
                data.append(x)
                n += 1
            else:
                print(msg)
        X = np.array(data)

        if b_mean is not None:
            X -= b_mean[None, :]

        # save the data

        fpath = osp.join(args.directory, args.label, f'{fname}.csv')
        dpath = osp.dirname(fpath)
        dirs = []
        while dpath != '':
            dirs.append(dpath)
            dpath = osp.dirname(dpath)
        for dir in dirs[::-1]:
            if not osp.isdir(dir):
                os.mkdir(dir)
        pandas.DataFrame(X).to_csv(fpath, header=None, index=None)

    ser.close()