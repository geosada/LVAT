#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os, sys, time, math

IS_NHWC_or_1D = 'NHWC' # 'NHWC'/'1D'

class HandleImageDataNumpy(object):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset

        # image width,height
        if self.dataset == "MNIST":
            from tensorflow.examples.tutorials import mnist
            _h, _w, _c = 28,28,1
            img_size = _h*_w*_c # the canvas size    
            _l = 10
        elif self.dataset == "CIFAR10":
            _h, _w, _c = 32,32,3
            img_size = _h*_w*_c 
            _l = 10
        elif self.dataset == "SVHN":
            #import loadSVHNKingma as svhn
            #PCA_DIM = 768
            PCA_DIM = -1    # no compressed raw data
            #img_size = PCA_DIM # PCA
            _h, _w, _c = 32,32,3
            img_size = _h*_w*_c 
            _l = 10
        elif self.dataset == "KaggleBreastHistology":
            _h, _w, _c = 50,50,3
            img_size = _h*_w*_c 
            _l = 2
        elif self.dataset == "BreaKHis":
            _h, _w, _c = 460,700,3
            img_size = _h*_w*_c 
            _l = 2
        elif self.dataset == "Kyoto2006":
            from loadKyoto2006 import loadKyoto2006
            _h, _w, _c = None,None,None
            img_size = None # dummy
            _l = 2 
        else: sys.exit("invalid dataset")

        self.h = _h
        self.w = _w
        self.c = _c
        self.l = _l
        self.img_size   = img_size
        self.batch_size = batch_size


        if self.dataset == "MNIST":
            PATH_OF_MNIST = "D:/data/img/MNIST/"
            data_directory = PATH_OF_MNIST
            if not os.path.exists(data_directory): os.makedirs(data_directory)
            mnist_datasets = mnist.input_data.read_data_sets(data_directory, one_hot=True)
            dataset_train, dataset_test = mnist_datasets.train, mnist_datasets.test  # binarized (0-1) mnist data

            n_examples_train = dataset_train.images.shape[0]
            n_examples_test = dataset_test.images.shape[0]
         
        elif self.dataset == "CIFAR10":
            #from cifar10 import load_cifar10
            from keras.datasets import cifar10
            (data_train, labels_train), (data_test, labels_test) = cifar10.load_data() # [0-255] integer
            data_train = data_train / 255.
            data_test  = data_test  / 255.

            if IS_NHWC_or_1D == '1D':
                data_train,   data_test    = data_train.reshape((-1, img_size)), data_test.reshape((-1, img_size)) # NHWC to 1d
            data_train,   data_test    = data_train.astype(np.float32), data_test.astype(np.float32)
            labels_train, labels_test  = labels_train.reshape((-1, )), labels_test.reshape((-1, ))  # flatten
        
            # if normalized or zca-ed one is preferable, 
            #data_train, labels_train, data_test, labels_test = cifar10.loadCIFAR10( PATH_OF_CIFAR10, use_cache=True)
            labels_train = self._one_hot_encoded(labels_train, 10)
            labels_test  = self._one_hot_encoded(labels_test,  10)
            n_examples_train = len(data_train)
            n_examples_test  = len(data_test)
        
        elif self.dataset == "SVHN":
            from svhn import load_svhn, NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST
            # data_train.shape is (604388,3072) w/ extra and (73257,3072) w/o extra
            #data_train, labels_train, data_test, labels_test = svhn.loadSVHN(cutoffdim=PCA_DIM, use_cache=False, use_extra=False)
            (data_train, labels_train), (data_test, labels_test) = load_svhn()
            labels_train = self._one_hot_encoded(labels_train, 10)
            labels_test  = self._one_hot_encoded(labels_test,  10)
            """
            n_examples_train = NUM_EXAMPLES_TRAIN
            n_examples_test  = NUM_EXAMPLES_TEST
        
            """
            n_examples_train = (data_train.shape[0]//self.batch_size) * self.batch_size  # discard residual
            data_train, labels_train = data_train[0:n_examples_train, :], labels_train[0:n_examples_train, :]
        
            n_examples_test = data_test.shape[0]//self.batch_size * self.batch_size
            data_test, labels_test = data_test[0:n_examples_test, :], labels_test[0:n_examples_test, :]
        
        elif self.dataset == "KaggleBreastHistology":
            from HandleIIDDataTFRecord import HandleIIDDataTFRecord
            K = 10
            TEST_IDXES = [9]
            PATHS = ( ['D:/data/img/KaggleBreastHistology'], None)  
            d = HandleIIDDataTFRecord( self.dataset, self.batch_size, K, PATHS, is_debug=False)

            (data_train, labels_train), (data_test, labels_test) = d.get_ndarrays(TEST_IDXES)
            labels_train = self._one_hot_encoded(labels_train, self.l)
            labels_test  = self._one_hot_encoded(labels_test,  self.l)


            #print('x:', data_train[0])
            #print('y:', labels_train[0])
            #sys.exit('kokomade')
            n_examples_train = len(data_train)
            n_examples_test  = len(data_test)

            n_examples_train = (data_train.shape[0]//self.batch_size) * self.batch_size  # discard residual
            n_examples_test = data_test.shape[0]//self.batch_size * self.batch_size
        
        elif self.dataset == "BreaKHis":
            from HandleIIDDataTFRecord import HandleIIDDataTFRecord
            K = 10
            TEST_IDXES = [9]
            PATHS = ( ['D:/data/img/BreaKHis/BreaKHis_v1/histology_slides/breast'], None)  
            d = HandleIIDDataTFRecord( self.dataset, self.batch_size, K, PATHS, is_debug=False)

            (data_train, labels_train), (data_test, labels_test) = d.get_ndarrays(TEST_IDXES)
            labels_train = self._one_hot_encoded(labels_train, self.l)
            labels_test  = self._one_hot_encoded(labels_test,  self.l)

            n_examples_train = len(data_train)
            n_examples_test  = len(data_test)

            #n_examples_train = (data_train.shape[0]//self.batch_size) * self.batch_size  # discard residual
            #data_train, labels_train = data_train[0:n_examples_train, :], labels_train[0:n_examples_train, :]
        
            #n_examples_test = data_test.shape[0]//self.batch_size * self.batch_size
            #data_test, labels_test = data_test[0:n_examples_test, :], labels_test[0:n_examples_test, :]
        
        elif self.dataset == "Kyoto2006":
            data_train, labels_train  = loadKyoto2006('train', use_sval=False, use_cache=True, as_onehot=True)
            data_test, labels_test    = loadKyoto2006( 'test', use_sval=False, use_cache=True, as_onehot=True)

            print(data_train.shape, labels_train.shape)
            n_examples_train = (data_train.shape[0]//self.batch_size) * self.batch_size  # discard residual
            data_train, labels_train = data_train[0:n_examples_train, :], labels_train[0:n_examples_train, :]
        
            n_examples_test = data_test.shape[0]//self.batch_size * self.batch_size
            data_test, labels_test = data_test[0:n_examples_test, :], labels_test[0:n_examples_test, :]

            self.img_size = data_train.shape[1]
            # ugly work waround for ImageInterface
            self.h, self.w, self.c = 1,1,self.img_size

        if self.dataset == "SVHN":
            pass
        else:
            assert(n_examples_train%self.batch_size ==0)
            assert(n_examples_test%self.batch_size ==0)
 
        if self.dataset == "MNIST":
            # following two properties are for MNIST.
            self.dataset_train  = dataset_train
            self.dataset_test   = dataset_test

            # bellow is just trial for crafting adv examples in eval.py
            self.data_train, self.labels_train = dataset_train.next_batch(55000) # x: (BATCH_SIZE x img_size)
            self.data_test,  self.labels_test  = dataset_test.next_batch(10000) # x: (BATCH_SIZE x img_size)

        else:
            self.data_train   = data_train
            self.labels_train = labels_train
            self.data_test    = data_test
            self.labels_test  = labels_test

        #if IS_NHWC_or_1D == 'NHWC':
        #    self.dataset_train = np.reshape( self.dataset_train, (self.batch_size, self.h, self.w, self.c))
        #    self.dataset_test  = np.reshape( self.dataset_test,  (self.batch_size, self.h, self.w, self.c))

        self.n_examples_train = n_examples_train
        self.n_examples_test  = n_examples_test 
        
        self.n_batches_train = int( self.n_examples_train/self.batch_size )
        self.n_batches_test  = int( self.n_examples_test/self.batch_size  )
        print('n_examples_train:%d, n_batches_train:%d, n_batches_test:%d' % \
                (self.n_examples_train, self.n_batches_train, self.n_batches_test))
        
    # DataHandler
    def _get_a_batch(self, data, labels, i):
        # data and labels should be (BATCH_SIZE, ..., ...)
        _batch_size = self.batch_size
        batch_data   = data[ i*_batch_size:(i+1)*_batch_size]
        batch_labels = labels[ i*_batch_size:(i+1)*_batch_size]
        return batch_data, batch_labels
    
    def _get_a_batch_old(self, data, labels, step):
        #print(data.shape)
        _batch_size = self.batch_size
        size = labels.shape[0]
        offset = (step * _batch_size) % (size - _batch_size)
        batch_data = data[offset:(offset + _batch_size), ...]
        batch_labels = labels[offset:(offset + _batch_size)]
        return batch_data, batch_labels
    
    def _one_hot_encoded(self, class_numbers, num_classes=None):
        # https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/06_CIFAR-10.ipynb
        # Find the number of classes if None is provided.
        if num_classes is None: num_classes = np.max(class_numbers) - 1
        return np.eye(num_classes, dtype=float)[class_numbers]

    def get_next_batch(self, i, is_train):
        _batch_size = self.batch_size
        if is_train:
            if self.dataset == "MNIST":
                x,y = self.dataset_train.next_batch(_batch_size) # x: (BATCH_SIZE x img_size)
            else:
                x,y = self._get_a_batch(self.data_train, self.labels_train, i )
        else:
            if self.dataset == "MNIST":
                x,y = self.dataset_test.next_batch(_batch_size) # x: (BATCH_SIZE x img_size)
            else:
                x,y = self._get_a_batch(self.data_test, self.labels_test, i )

        if IS_NHWC_or_1D == 'NHWC':
            x = np.reshape( x, (_batch_size, self.h, self.w, self.c))

        return x,y


############################
"""       MISC           """
############################
class utils(object):
    def list2str(l): return ", ".join (map(str,l))
    def list2mu(l,i, stepback=1): return np.mean(np.array(l[i])/ np.array(l[i-stepback]))

if __name__ == '__main__':
    BATH_SIZE = 50
    DATASET = 'MNIST'
    DATASET = 'CIFAR10'
    DATASET = 'SVHN'
    DATASET = 'BreaKHis'
    d = HandleImageDataNumpy(DATASET, BATH_SIZE)
    _x, _y = d.get_next_batch(3, True)
    print(_x, _y)

