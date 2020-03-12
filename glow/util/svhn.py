from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from scipy.io import loadmat

import numpy as np
from scipy import linalg
import glob
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf
from dataset_utils import *

DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

N_LABELED = 73257
#N_LABELED = 1000
DATASET_SEED = 1
#DATA_DIR = '/data/img/SVHN__labled_1000'
DATA_DIR = '/data/img/SVHN'
DATA_DIR = 'D:/data/img/SVHN'

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_string('data_dir', '/tmp/svhn', "")
#tf.app.flags.DEFINE_integer('num_labeled_examples', 1000, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_valid_examples', 1000, "The number of validation examples")
#tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")

NUM_EXAMPLES_TRAIN = 73257
NUM_EXAMPLES_TEST = 26032


def maybe_download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    filepath_train_mat = os.path.join(DATA_DIR, 'train_32x32.mat')
    filepath_test_mat = os.path.join(DATA_DIR, 'test_32x32.mat')
    print(filepath_train_mat)
    print(filepath_test_mat)
    if not os.path.exists(filepath_train_mat) or not os.path.exists(filepath_test_mat):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(DATA_URL_TRAIN, filepath_train_mat, _progress)
        urllib.request.urlretrieve(DATA_URL_TEST, filepath_test_mat, _progress)

    # Training set
    print("... loading training data")
    train_data = loadmat(DATA_DIR + '/train_32x32.mat')

    # geosada 170717
    #train_x = (-127.5 + train_data['X']) / 255.
    train_x = (train_data['X']) / 255.
    train_x = train_x.transpose((3, 0, 1, 2))
    train_x = train_x.reshape([train_x.shape[0], -1])
    train_y = train_data['y'].flatten().astype(np.int32)
    train_y[train_y == 10] = 0

    # Test set
    print("... loading testing data")
    test_data = loadmat(DATA_DIR + '/test_32x32.mat')
    # geosada 170717
    #test_x = (-127.5 + test_data['X']) / 255.
    test_x = (test_data['X']) / 255.
    test_x = test_x.transpose((3, 0, 1, 2))
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = test_data['y'].flatten().astype(np.int32)
    test_y[test_y == 10] = 0

    print("... saving npy as cache")
    np.save('{}/train_images'.format(DATA_DIR), train_x)
    np.save('{}/train_labels'.format(DATA_DIR), train_y)
    np.save('{}/test_images'.format(DATA_DIR), test_x)
    np.save('{}/test_labels'.format(DATA_DIR), test_y)


def load_svhn(_use_cache=False):

    if _use_cache:
         print("... loading data from npy cache")
    else:
        maybe_download_and_extract()

    #
    #   returned shape:
    #       images: (n, img_size)
    #       labels: (n,)
    #
    train_images = np.load('{}/train_images.npy'.format(DATA_DIR)).astype(np.float32)
    train_labels = np.load('{}/train_labels.npy'.format(DATA_DIR)).astype(np.uint8)
    test_images = np.load('{}/test_images.npy'.format(DATA_DIR)).astype(np.float32)
    test_labels = np.load('{}/test_labels.npy'.format(DATA_DIR)).astype(np.uint8)
    return (train_images, train_labels), (test_images, test_labels)


def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = load_svhn()
    dirpath = os.path.join(DATA_DIR, 'seed' + str(DATASET_SEED))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    rng = np.random.RandomState(DATASET_SEED)
    rand_ix = rng.permutation(NUM_EXAMPLES_TRAIN)
    _train_images, _train_labels = train_images[rand_ix], train_labels[rand_ix]


    if N_LABELED == NUM_EXAMPLES_TRAIN:
        print('>>> superviesed <<<')
        labeled_ind = np.arange(N_LABELED)
        labeled_train_images, labeled_train_labels = _train_images[labeled_ind], _train_labels[labeled_ind]
        _train_images = np.delete(_train_images, labeled_ind, 0)
        _train_labels = np.delete(_train_labels, labeled_ind, 0)

    else:
        print('>>> semi-superviesed <<<')
        examples_per_class = int(N_LABELED / 10)
        labeled_train_images = np.zeros((N_LABELED, 3072), dtype=np.float32)
        labeled_train_labels = np.zeros((N_LABELED), dtype=np.int64)
        for i in xrange(10):
            ind = np.where(_train_labels == i)[0]
            labeled_train_images[i * examples_per_class:(i + 1) * examples_per_class] \
                = _train_images[ind[0:examples_per_class]]
            labeled_train_labels[i * examples_per_class:(i + 1) * examples_per_class] \
                = _train_labels[ind[0:examples_per_class]]
            _train_images = np.delete(_train_images,
                                         ind[0:examples_per_class], 0)
            _train_labels = np.delete(_train_labels,
                                         ind[0:examples_per_class])
                                                                                                                                  
        rand_ix_labeled = rng.permutation(N_LABELED)
        labeled_train_images, labeled_train_labels = labeled_train_images[rand_ix_labeled], labeled_train_labels[rand_ix_labeled]
                                                                                                                                  
        #print(labeled_train_images.shape, labeled_train_labels.shape)
        #print(train_images.shape, train_labels.shape)
        #print(test_images.shape, test_labels.shape)
        #print(labeled_train_labels)

    convert_images_and_labels(labeled_train_images,
                              labeled_train_labels,
                              os.path.join(dirpath, 'labeled_train.tfrecords'))
    convert_images_and_labels(train_images, train_labels,
                              os.path.join(dirpath, 'unlabeled_train.tfrecords'))
    convert_images_and_labels(test_images,
                              test_labels,
                              os.path.join(dirpath, 'test.tfrecords'))

    # Construct dataset for validation
    train_images_valid, train_labels_valid = labeled_train_images, labeled_train_labels
    test_images_valid, test_labels_valid = \
        _train_images[:FLAGS.num_valid_examples], _train_labels[:FLAGS.num_valid_examples]
    unlabeled_train_images_valid = np.concatenate(
        (train_images_valid, _train_images[FLAGS.num_valid_examples:]), axis=0)
    unlabeled_train_labels_valid = np.concatenate(
        (train_labels_valid, _train_labels[FLAGS.num_valid_examples:]), axis=0)
    convert_images_and_labels(train_images_valid,
                              train_labels_valid,
                              os.path.join(dirpath, 'labeled_train_val.tfrecords'))
    convert_images_and_labels(unlabeled_train_images_valid,
                              unlabeled_train_labels_valid,
                              os.path.join(dirpath, 'unlabeled_train_val.tfrecords'))
    convert_images_and_labels(test_images_valid,
                              test_labels_valid,
                              os.path.join(dirpath, 'test_val.tfrecords'))


def inputs(batch_size=100,
           train=True, validation=False,
           shuffle=True, num_epochs=None):
    if validation:
        if train:
            filenames = ['labeled_train_val.tfrecords']
            num_examples = N_LABELED
        else:
            filenames = ['test_val.tfrecords']
            num_examples = FLAGS.num_valid_examples
    else:
        if train:
            filenames = ['labeled_train.tfrecords']
            num_examples = N_LABELED
        else:
            filenames = ['test.tfrecords']
            num_examples = NUM_EXAMPLES_TEST

    filenames = [os.path.join('seed' + str(DATASET_SEED), filename) for filename in filenames]
    filename_queue = generate_filename_queue(filenames, DATA_DIR, num_epochs)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32)) if train else image
    return generate_batch([image, label], num_examples, batch_size, shuffle)


def unlabeled_inputs(batch_size=100,
                     validation=False,
                     shuffle=True):
    if validation:
        filenames = ['unlabeled_train_val.tfrecords']
        num_examples = NUM_EXAMPLES_TRAIN - FLAGS.num_valid_examples
    else:
        filenames = ['unlabeled_train.tfrecords']
        num_examples = NUM_EXAMPLES_TRAIN

    filenames = [os.path.join('seed' + str(DATASET_SEED), filename) for filename in filenames]
    filename_queue = generate_filename_queue(filenames, data_dir=DATA_DIR)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32))
    return generate_batch([image], num_examples, batch_size, shuffle)


def main(argv):
    prepare_dataset()


if __name__ == "__main__":
    tf.app.run()
