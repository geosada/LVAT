# Copyright 2015 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from scipy import linalg
import glob
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf

from dataset_utils import *

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

#tf.app.flags.DEFINE_string('data_dir', 'C:/Users/fx29351/Python/data/CIFAR10', 'where to store the dataset')
#tf.app.flags.DEFINE_integer('num_labeled_examples', 4000, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_valid_examples', 1000, "The number of validation examples")
#tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")
FLAGS = tf.app.flags.FLAGS

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

N_LABELED = 4000
DATASET_SEED = 1
DATA_DIR = 'D:/data/img/CIFAR10'

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 50000
NUM_EXAMPLES_TEST = 10000

def load_cifar10():
    """Download and extract the tarball from Alex's website."""
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    # Training set
    print("Loading training data...")
    train_images = np.zeros((NUM_EXAMPLES_TRAIN, 3 * 32 * 32), dtype=np.float32)
    train_labels = []
    for i, data_fn in enumerate(
            sorted(glob.glob(DATA_DIR + '/cifar-10-batches-py/data_batch*'))):
        batch = unpickle(data_fn)
        train_images[i * 10000:(i + 1) * 10000] = batch['data']
        train_labels.extend(batch['labels'])

    # geosada 170713 for generative model
    #train_images = (train_images - 127.5) / 255.
    # -> [0,1]
    train_images = train_images / 255.
    train_labels = np.asarray(train_labels, dtype=np.int64)

    rand_ix = np.random.permutation(NUM_EXAMPLES_TRAIN)
    train_images = train_images[rand_ix]
    train_labels = train_labels[rand_ix]

    print("Loading test data...")
    test = unpickle(DATA_DIR + '/cifar-10-batches-py/test_batch')
    test_images = test['data'].astype(np.float32)
    # geosada 170713
    #test_images = (test_images - 127.5) / 255.
    # -> [0,1]
    test_images = test_images / 255.
    test_labels = np.asarray(test['labels'], dtype=np.int64)

    # geosada 170713 for generative model
    """
    print("Apply ZCA whitening")
    components, mean, train_images = ZCA(train_images)
    np.save('{}/components'.format(DATA_DIR), components)
    np.save('{}/mean'.format(DATA_DIR), mean)
    test_images = np.dot(test_images - mean, components.T)
    """

    train_images = train_images.reshape(
        (NUM_EXAMPLES_TRAIN, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TRAIN, -1))
    test_images = test_images.reshape(
        (NUM_EXAMPLES_TEST, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TEST, -1))
    return (train_images, train_labels), (test_images, test_labels)


def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()

    dirpath = os.path.join(DATA_DIR, 'seed' + str(DATASET_SEED))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    rng = np.random.RandomState(DATASET_SEED)
    rand_ix = rng.permutation(NUM_EXAMPLES_TRAIN)
    _train_images, _train_labels = train_images[rand_ix], train_labels[rand_ix]

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
    labeled_train_images, labeled_train_labels = \
        labeled_train_images[rand_ix_labeled], labeled_train_labels[rand_ix_labeled]

    convert_images_and_labels(labeled_train_images,
                              labeled_train_labels,
                              os.path.join(dirpath, 'labeled_train.tfrecords'))
    convert_images_and_labels(train_images, train_labels,
                              os.path.join(dirpath, 'unlabeled_train.tfrecords'))
    convert_images_and_labels(test_images,
                              test_labels,
                              os.path.join(dirpath, 'test.tfrecords'))

    # Construct dataset for validation
    train_images_valid, train_labels_valid = \
        labeled_train_images[FLAGS.num_valid_examples:], labeled_train_labels[FLAGS.num_valid_examples:]
    test_images_valid, test_labels_valid = \
        labeled_train_images[:FLAGS.num_valid_examples], labeled_train_labels[:FLAGS.num_valid_examples]
    unlabeled_train_images_valid = np.concatenate(
        (train_images_valid, _train_images), axis=0)
    unlabeled_train_labels_valid = np.concatenate(
        (train_labels_valid, _train_labels), axis=0)
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
            num_examples = N_LABELED - FLAGS.num_valid_examples
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

    # geosada 170701
    filenames = [os.path.join('seed' + str(DATASET_SEED), filename) for filename in filenames]
    #filenames = ['C:/Users/fx29351/Python/data/CIFAR10/seed' + str(DATASET_SEED) + '/' + filename for filename in filenames]

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

    # geosada 170701
    filenames = [os.path.join('seed' + str(DATASET_SEED), filename) for filename in filenames]
    #filenames = ['C:/Users/fx29351/Python/data/CIFAR10/seed' + str(DATASET_SEED) + '/' + filename for filename in filenames]
    filename_queue = generate_filename_queue(filenames, DATA_DIR)
    image, label = read(filename_queue)
    image = transform(tf.cast(image, tf.float32))
    return generate_batch([image], num_examples, batch_size, shuffle)


def main(argv):
    prepare_dataset()


if __name__ == "__main__":
    tf.app.run()
