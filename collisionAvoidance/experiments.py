from __future__ import division
from pyd_net import *
import tensorflow.contrib.slim as slim
import tensorflow as tf
import time
import re
import argparse
import numpy as np

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--dataset',           type=str,
                    help='dataset to train on, lion, or cat', default='cat')
parser.add_argument('--datapath',          type=str,
                    help='path to the data', required=True)
parser.add_argument('--filenames',         type=str,
                    help='path to the filenames text file', required=True)
parser.add_argument('--output_directory',  type=str,
                    help='output directory for test disparities, if empty outputs to checkpoint folder', default='.')
parser.add_argument('--checkpoint_dir',        type=str,
                    help='path to a specific checkpoint to load', default='checkpoint/IROS18/pydnet')
parser.add_argument('--resolution',        type=int,
                    default=1, help='resolution [1:H, 2:Q, 3:E]')

args = parser.parse_args()


def read_image(image_path):
    image = tf.image.decode_image(tf.read_file(args.datapath+'/'+image_path))
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image,  tf.float32)
    image = tf.expand_dims(tf.image.resize_images(
        image,  [256, 512], tf.image.ResizeMethod.AREA), 0)

    return image


def test_code(params):

    input_queue = tf.train.string_input_producer(
        [args.filenames], shuffle=False)
    line_reader = tf.TextLineReader()
    _, line = line_reader.read(input_queue)
    img_path = tf.string_split([line]).values[0]
    img = read_image(img_path)

    placeholders = {'im0': img}

    with tf.variable_scope("model") as scope:
        model = pydnet(placeholders)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    train_saver.restore(sess, args.checkpoint_dir)

    f = open(args.filenames, 'r')
    samples = len(f.readlines())
    f.close()

    print('Testing {} frames'.format(samples))
    disparities = np.zeros((samples, 256, 512), dtype=np.float32)
    for step in range(samples):
        print('Running %d out of %d' % (step, samples))

        # res 1 nad 2 for lower resolution
        disp = sess.run(model.results[args.resolution-1])
        disparities[step] = disp[0, :, :, 0].squeeze()

    print('Test done!')

    print('Saving disparities as .npy')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_dir)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy', disparities)

    print('Disparities saved!')
# main function


def main(_):

    test_code(args)


if __name__ == '__main__':
    tf.app.run()
