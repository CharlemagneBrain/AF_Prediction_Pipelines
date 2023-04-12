'''

The sample data shape is 
    h5py {
        'tracings': (N, 4096, 12), # (number_of_data, points, channels)
        
    }
If you want modify this for your own hdf5 data, 
the only thing you need to modify is "get_feature(point_cloud, label)" function
'''

import h5py
import tensorflow as tf


# For array storage, TFRecords will only support list storage or 1-D array storage
# If you have multi-dimensional array, please start with:
#     array = array.reshape(-1)
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_all_keys_from_h5(h5_file):
    res = []
    for key in h5_file.keys():
        res.append(key)
    return res

# details of 9-D vector: https://github.com/charlesq34/pointnet/issues/7
def get_feature(point_cloud):
    return {
        'DI': _floats_feature(point_cloud[:, 0]),
        'DII': _floats_feature(point_cloud[:, 1]),
        'DIII': _floats_feature(point_cloud[:, 2]),
        'AVR': _floats_feature(point_cloud[:, 3]),
        'AVL': _floats_feature(point_cloud[:, 4]),
        'AVF': _floats_feature(point_cloud[:, 5]),
        'V1': _floats_feature(point_cloud[:, 6]),
        'V2': _floats_feature(point_cloud[:, 7]),
        'V3': _floats_feature(point_cloud[:, 8]),
        'V4': _floats_feature(point_cloud[:, 9]),
        'V5': _floats_feature(point_cloud[:, 10]),
        'V6': _floats_feature(point_cloud[:, 11]),
        #'label': _int64_feature(label)
    }

def h5_to_tfrecord_converter(input_file_path, output_file_path):
    h5_file = h5py.File(input_file_path)
    keys = get_all_keys_from_h5(h5_file)
    
    num_of_items = h5_file[keys[0]][:].shape[0]

    # Check the number of values in each key
    for key in keys:
        if h5_file[key][:].shape[0] != num_of_items:
            raise ValueError('Invalid values. The inequality of the number of values in each key.')

    with tf.io.TFRecordWriter(output_file_path) as writer:
        for index in range(num_of_items):
            example = tf.train.Example(
              features=tf.train.Features(
                  feature = get_feature(h5_file[keys[0]][index])
              ))
            writer.write(example.SerializeToString())
            print('\r{:.1%}'.format((index+1)/num_of_items), end='')
    
# With commandline enabled
if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file-path', required=True, help='Path to the input HDF5 file.')
    parser.add_argument('--output-file-path', default='', help='Path to the output TFRecords.')
    parser.add_argument('-r', action='store_true', help='Recursively find *.h5 files under pointed folder. This will not dive deeper into sub-folders.')
    FLAGS = parser.parse_args()

    INPUT_PATH = os.path.abspath(FLAGS.input_file_path)
    OUTPUT_PATH = FLAGS.output_file_path
    RECURSIVE = FLAGS.r

    if not INPUT_PATH.endswith('.h5') and not RECURSIVE:
        raise ValueError('Not a valid HDF5 file provided, you may want to add -r.')

    elif INPUT_PATH.endswith('.h5'):
        if OUTPUT_PATH == '':
            OUTPUT_PATH = INPUT_PATH[:-3]
        print('Start converting...\t')
        h5_to_tfrecord_converter(INPUT_PATH, os.path.abspath(OUTPUT_PATH) + '.tfrecord')

    elif RECURSIVE:
        files = []
        if OUTPUT_PATH == '':
            OUTPUT_PATH = INPUT_PATH
        for _file in os.listdir(INPUT_PATH):
            if _file.endswith('.h5'):
                files.append((
                    os.path.join(INPUT_PATH, _file[:-3]), 
                    os.path.join(os.path.abspath(OUTPUT_PATH), _file[:-3]),
                    _file
                    ))
        print(len(files), 'of HDF5 file detected.')
        for idx, (_input, _output, _file_name) in enumerate(files):
            print('\n\ton job %d/%d, %s' % (idx, len(files), _file_name), end='')
            h5_to_tfrecord_converter(_input + '.h5', _output + '.tfrecord')

    else:
        pass
        