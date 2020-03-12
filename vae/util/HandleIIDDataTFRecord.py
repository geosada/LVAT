import numpy as np
import pandas as pd
import tensorflow as tf
import sys, os, time
from tqdm import tqdm

import config as c

#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')


_label_list_local = []

IS_UNLABELED_ENABLE = False

class HandleIIDDataTFRecord(object):

    def __init__(self, is_debug=False):

        """ for MNIST, SVHN, and CIFAR10, k and path_root are dummy """

        self._k         = c.K 
        self.is_debug   = is_debug

        if c.FLAGS.dataset == 'CIFAR10':
            n_train, n_test = 50000, 10000
            _h, _w, _c = 32,32,3
            _img_size = _h*_w*_c
            _l = 10
        elif c.FLAGS.dataset == 'SVHN':
            from svhn import NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST, N_LABELED	
            n_train, n_test = 73257, 26032
            _h, _w, _c = 32,32,3
            _img_size = _h*_w*_c
            _l = 10
        elif c.FLAGS.dataset == 'MNIST':
            n_train, n_test = 55000, 10000
            _h, _w, _c = 28,28,1
            _img_size = _h*_w*_c
            _l = 10

        elif c.FLAGS.dataset == 'KaggleBreastHistology':
            from loadKaggleBreastHistology import _label_list
            global _label_list_local
            _label_list_local = _label_list

            n_train, n_test = 0,0
            _h, _w, _c = 50,50,3
            _img_size = _h*_w*_c
            _l = len(_label_list_local)
            self.is_x_filepath = False

            self.path_root_trains, self.path_root_tests = c.PATHS
            assert len(self.path_root_trains)==1, 'invalid'

            base = self.path_root_trains[0] + '/' + 'cache_%s'%(c.FLAGS.dataset)
            self.file_cache_x = base + '_x_%d'
            self.file_cache_y = base + '_y_%d'

            self.paths_rawimage_train = self.path_root_trains
            self.paths_tfrecord_train = [ a + '/TFRECORD/' for a in self.path_root_trains]

        """
        elif c.FLAGS.dataset == 'BreaKHis':
            from loadBreaKHis import _label_list
            global _label_list_local
            _label_list_local = _label_list

            n_train, n_test = 0,0
            _h, _w, _c = 460,700,3
            _img_size = _h*_w*_c
            _l = len(_label_list_local)
            self.is_x_filepath = True

            self.path_root_trains, self.path_root_tests = _paths
            assert len(self.path_root_trains)==1, 'invalid'
            
            base = self.path_root_trains[0] + '/' + 'cache_%s'%(c.FLAGS.dataset)
            self.file_cache_x = base + '_x_%d'
            self.file_cache_y = base + '_y_%d'

            self.paths_rawimage_train = self.path_root_trains
            self.paths_tfrecord_train = [ a + '/TFRECORD/' for a in self.path_root_trains]
            if self.path_root_tests is not None:
                self.path_tfrecord_test = [ a + '/TFRECORD/' for a in self.path_root_tests]

        elif c.FLAGS.dataset == 'CharImages':
            from loadCharImages import _label_list
            global _label_list_local
            _label_list_local = _label_list

            n_train, n_test = 0,0
            _h, _w, _c = 32,32,1
            _img_size = _h*_w*_c
            _l = len(_label_list_local)
            self.is_x_filepath = True

            self.path_root_trains, self.path_root_tests = _paths
            self.paths_rawimage_train = [ a + '/PNG/'      for a in self.path_root_trains]
            self.paths_tfrecord_train = [ a + '/TFRECORD/' for a in self.path_root_trains]
            if self.path_root_tests is not None:
                self.path_tfrecord_test = [ a + '/TFRECORD/' for a in self.path_root_tests]
        """


        self.h = _h
        self.w = _w
        self.c = _c
        self.l = _l
        self.img_size  = _img_size
        self.n_train   = n_train
        self.n_test    = n_test

        # 190827 modification along vat(SSL)
        #self.n_batches_train = int(n_train/batch_size)
        #self.n_batches_test  = int(n_test/batch_size)
        self.n_batches_train  = int( n_train/ c.BATCH_SIZE)  
        self.n_batches_test   = int( n_test / c.BATCH_SIZE_TEST)

    ########################################
    """             inputs              """
    ########################################
    def get_ndarrays(self, idxes):

        if not ( c.FLAGS.dataset == 'KaggleBreastHistology' or
                 c.FLAGS.dataset == 'BreaKHis' or
                 c.FLAGS.dataset == 'CharImages'):
            sys.exit('get_ndarrays() has not been implemented yet for %s'%(c.FLAGS.dataset))
            
        indices_train, indices_test = _split_k_into_train_and_test(self._k, idxes)

        def read_npy(filename, idxes):
            files = []
            files.extend( [ filename%(i)+'.npy' for i in idxes ])

            o = []
            """
            for file in files:
                print('[INFO] ... loading', file)
                data = np.load(file)
                o.append( data)
            return np.vstack(o)
            """
            for file in files:
                print('[INFO] ... loading', file)
                data = np.load(file)
                o.extend( data)
            o = np.array(o)
            return o

        x_train = read_npy(self.file_cache_x, indices_train)
        y_train = read_npy(self.file_cache_y, indices_train)
        x_test  = read_npy(self.file_cache_x, indices_test)
        y_test  = read_npy(self.file_cache_y, indices_test)
        return (x_train, y_train), (x_test, y_test)

    def get_tfrecords(self, idxes=None):

        """
        idxes: idxes of TEST set in k-fold cross varidation. i.e., list of i, where 0 <= i < k+1
        xtrain: all records
        *_l   : partial records
        """
        if c.FLAGS.dataset == 'CIFAR10':
            from cifar10 import inputs, unlabeled_inputs
            xtrain_l, ytrain_l = inputs(batch_size=c.BATCH_SIZE, train=True,  validation=False, shuffle=True)
            xtrain             = unlabeled_inputs(batch_size=c.BATCH_SIZE_UL,    validation=False, shuffle=True)
            xtest , ytest      = inputs(batch_size=c.BATCH_SIZE_TEST, train=False, validation=False, shuffle=True)
        elif c.FLAGS.dataset =='SVHN':
            from svhn import inputs, unlabeled_inputs
            xtrain_l, ytrain_l = inputs(batch_size=c.BATCH_SIZE, train=True,  validation=False, shuffle=True)
            xtrain             = unlabeled_inputs(batch_size=c.BATCH_SIZE_UL,    validation=False, shuffle=True)
            xtest , ytest      = inputs(batch_size=c.BATCH_SIZE_TEST, train=False, validation=False, shuffle=True)
        elif c.FLAGS.dataset == 'MNIST':
            from mnist import inputs
            xtrain_l, ytrain_l = inputs(c.BATCH_SIZE, 'train_labeled')
            xtrain,_           = inputs(c.BATCH_SIZE_UL, 'train')
            xtest , ytest      = inputs(c.BATCH_SIZE_TEST, 'test')

        else:
            indices_train, indices_test = _split_k_into_train_and_test(self._k, idxes)
            #indices_train = [0]
            #print('=====================================================')
            #print('CAUTION! n of tfrecord_train was forced to be reduced', indices_train)
            #print('=====================================================')

            paths_train = self.paths_tfrecord_train

            if idxes is None:
                """ use different data sources for training and test respectively."""
                if self.path_root_tests is None:
                    raise ValueError('path_root_tests was not given.')
                path_tests = self.path_tfrecord_test

            else:
                """ do cross validataion over a single data source. """

                if self.path_root_tests is not None:
                    raise ValueError('selection of test data source is confusing since both path_root_tests and idxes are given.')

                paths_test = self.paths_tfrecord_train
                                                                                                             
            """ DEBUG
            print('indices_train:',indices_train)
            print('indices_test:', indices_test)
            sys.exit('oshimai')
            """

            print('... reading TFRecords for train')
            (xtrain_l, ytrain_l), n_train_l = self.inputs(paths_train, indices_train, batch_size=c.BATCH_SIZE)
                                                                                                             
            print('... reading TFRecords for test')
            (xtest, ytest), n_test = self.inputs(paths_test, indices_test, batch_size=c.BATCH_SIZE_TEST)
                                                                                                             
            if IS_UNLABELED_ENABLE:
                print('... reading TFRecords of unlabeled')
                (xtrain, _), n_train_u = self.inputs(paths_train, indices_all, batch_size=c.BATCH_SIZE_UL )
            else:
                xtrain = xtrain_l
                n_train_u = 0
                                                                                                             
            """ CAUTION!
                in SSL setting, x_l and x_ul are in parallel and independently input to the model.
                bellow n_batches_train would be used as the num of training iteration per epoch,
                but when c.BATCH_SIZE and c.BATCH_SIZE_UL are different, n_batches_train is no longer valid.
                so, instead of n_batches_train, the num of training iteration per epoch should be given as arg, as vat original code.
            """
            #self.n_batches_train  = int((n_train_l + n_train_u) / (c.BATCH_SIZE + c.BATCH_SIZE_UL))
            self.n_batches_train  = int((n_train_l + n_train_u) / (c.BATCH_SIZE ))

            self.n_batches_test   = int( n_test / c.BATCH_SIZE_TEST)
            print(' n_batches_train: %d, # n_batches_test : %d'%(self.n_batches_train, self.n_batches_test))
            # [ToDo] the way to set n_labeled is inconsistent in datasets.
            self.n_labeled        = n_train_l
            self.n_train          = n_train_l + n_train_u

        return (xtrain_l, ytrain_l), xtrain, (xtest , ytest)

    def inputs(self, paths, indices, batch_size, shuffle=True, num_epochs=None):


        def generate_batch( example, shuffle):
            #print(example)
            sequence, label = example
            num_preprocess_threads = 1
            min_queue_examples = 100
            
            if shuffle:
                ret = tf.train.shuffle_batch( [sequence, label], batch_size=batch_size,
                        num_threads=4, capacity=5000, min_after_dequeue=10)
            else:
                ret = tf.train.batch( [sequence, label], batch_size=batch_size, num_threads=num_preprocess_threads,
                    allow_smaller_final_batch=True, capacity=min_queue_examples + 5 * batch_size)
            print('exit generate_batch')
            return ret
                                                                                                                       
        def read(num_epochs=None):

            tfrecords = []
            for path in paths:
                tfrecords.extend( [ path + 'train_%d.tfrecord'%(i) for i in indices ])
            #print(tfrecords)
            #sys.exit('oshimai')

            
            ###########################################
            """         inspect tfrecords           """
            ###########################################
            n_examples = 0
            for file in tfrecords:
                if not os.path.exists(file):
                    sys.exit('[ERROR] %s was not found.'%(file))
                c = 0
                for record in tf.python_io.tf_record_iterator(file):
                    c += 1
                n_examples += c
                print('... %d records in %s'%(c, file))
            print('... in total:', n_examples)

            ###########################################
            """           read tfrecords           """
            ###########################################
            reader = tf.TFRecordReader()
            queue = tf.train.string_input_producer(tfrecords, num_epochs=num_epochs)
            _, serialized_example = reader.read(queue)

            _h,_w,_c = self.h, self.w, self.c
            features = tf.parse_single_example(
                serialized_example,
                # Defaults are not specified since both keys are required.
                features={
                    'image': tf.FixedLenFeature([_h*_w*_c], tf.float32),
                    'label': tf.FixedLenFeature([], tf.int64),
                })

            # Convert label from a scalar uint8 tensor to an int32 scalar.
            images = features['image']
            images = tf.reshape(images, [_h,_w,_c])
            labels = tf.one_hot(tf.cast(features['label'], tf.int32), self.l)

            return images, labels, n_examples
            """ end of read() """

        sequence, label, n_examples = read()
        return generate_batch([sequence, label], shuffle), n_examples


    ########################################
    """             prepare             """
    ########################################
    def prepare(self, do_write_tfrecord, do_write_npy):

        if c.FLAGS.dataset == 'CIFAR10' or c.FLAGS.dataset == 'SVHN':
            sys.exit('[ERROR] execute each utility such as mnist.py and svhn.py.')

        elif c.FLAGS.dataset == 'MNIST':

            from mnist import prepare_dataset
            prepare_dataset(is_process_only_labeled_data=False)

        else:

            if c.FLAGS.dataset == 'KaggleBreastHistology':

                from loadKaggleBreastHistology import loadKaggleBreastHistology
                loader = loadKaggleBreastHistology

            elif c.FLAGS.dataset == 'BreaKHis':

                from loadBreaKHis import loadBreaKHis
                loader = loadBreaKHis

            elif c.FLAGS.dataset == 'CharImages':

                from loadCharImages import loadCharImages
                loader = loadCharImages

            for path_rawimage, path_tfrecord in zip(self.paths_rawimage_train, self.paths_tfrecord_train):

                print('... start preparation under:', path_rawimage)
                data = loader( path_rawimage, is_debug=self.is_debug )

                data = _divide_into_k_folds(data, self._k, self.l)

                if do_write_tfrecord:
                    self.write_tfrecord(path_tfrecord, data)

                if do_write_npy:
                    self.write_npy( data)
        return data

    def preprocess(self, x):
        return x/255.

    def read_image(self, filenames):
        import scipy.misc

        o = []
        shape = (self.h, self.w, self.c)
        #for filename in tqdm(filenames):
        for filename in filenames:
           
            #print('filename:', filename)
            data = np.array(Image.open(filename))
            if data.shape != shape:
                print('[WARN] iregular size image was found. it will be resized: ', filename)
                data = scipy.misc.imresize(data, shape)

            data = data[np.newaxis, ...]
            o.append(data)
                                                  
        return np.vstack(o)

    def write_npy(self,_data ):

        for i in range(self._k):

            print('[INFO] ... reading images', i+1, '/' ,self._k)
            
            if self.is_x_filepath:
                x = self.read_image(_data[i]['Image'].tolist())
            else:
                x = _data[i]['Image']

            x = self.preprocess(x)

            print('[INFO] ... writing', self.file_cache_x%(i))
            np.save(self.file_cache_x%(i), x)
            del x
            y =  _data[i]['Label_ID'].tolist()
            print('[INFO] ... writing', self.file_cache_y%(i))
            np.save(self.file_cache_y%(i), y)

    def write_tfrecord(self, path_tfrecord, _data ):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def make_example( image, label):
            #print('image:', image)
            #print('label:', label)
            #sys.exit('aaa')

            _h,_w,_c = self.h, self.w, self.c
            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'height': _int64_feature(_h),
                        'width':  _int64_feature(_w),
                        'depth':  _int64_feature(_c),
                        'label':  _int64_feature(int(label)),
                        'image':  tf.train.Feature(
                            float_list=tf.train.FloatList(value=image)
                        )
            }))
            return example

        def write( filename, image_files, labels):
            """
            image_files: full path of img files to be read
            """
            writer = tf.python_io.TFRecordWriter(filename)
            print('... writing %s, length: %d %d'%(filename, len(image_files), len(labels)))

            shape = (self.h, self.w, self.c)
            for image_file, label in tqdm(zip(image_files, labels), total=len(labels)):
               
                """
                x = np.array(Image.open(image_file))
                if x.shape != shape:
                    print('[WARN] iregular size image was found. it will be resized: ', image_file)
                    x = scipy.misc.imresize(x, shape)
                """

                # kokokara 2/19
                #x = self.read_image([image_file])
                if self.is_x_filepath:
                    x = self.read_image([image_file])
                else:
                    x = image_file
                
                x = self.preprocess(x)


                #
                # geosada 190220
                #
                x = x.tostring()
                ex = make_example(x, label)
                writer.write(ex.SerializeToString())
            writer.close()

        for i in range(self._k):
            #for path_tfrecord in self.paths_tfrecord_train:
            filename = path_tfrecord  + 'train_%d.tfrecord'%(i)
            write( filename, _data[i]['Image'].tolist(),  _data[i]['Label_ID'].tolist() )

        """
        if IS_UNLABELED_ENABLE:
            filename = self.paths_tfrecord_train + 'train_unlabeled.tfrecords'
            do_write( filename, unlabeled['Image'].tolist(),  unlabeled['Label_ID'].tolist() )
        """

def _split_k_into_train_and_test(_k, idxes):

    idxes_train, idxes_test = [],[]

    if idxes is None:
        print('[WARN] idxes for test in ', _k, 'folds was given as None.')
        idxes_train = list(range(_k))
        return idxes_train, idxes_test

    for i in range(_k):
        if i in idxes:
            idxes_test.append(i)
        else:
            idxes_train.append(i)
    return idxes_train, idxes_test

def _divide_into_k_folds(_data, _k, _l):
    """
    input:
        _data: dict( label_name: [list_of full-path for raw image file or ndarray])
        
    _k: n of k-folds
    _l: n of classes
    return:
        [list of _k DataFrame consisting of 2 colmuns]:
            'Image': full-path for raw image file
            'Label_ID' : label
    """
    #
    # convert dict to DataFrame
    #
    data_labels, data_images = [],[]
    for label, images in _data.items():
        labels = [label] * len(images)
        data_labels.extend(labels)
        data_images.extend(images)
    _data = pd.DataFrame({'Label_ID' : data_labels, 'Image': data_images})


    
    #
    # split for cross validation
    #
    o = [pd.DataFrame()] * _k

    for y in range(_l):

        a_family = _data[_data['Label_ID']==y ] 
        #a_family = a_family.reset_index(drop=True)
        dfs = np.array_split(a_family, _k) # divide roughly equally
        for i in range(_k):
            o[i] = pd.concat((o[i], dfs[i]))
    return o



if __name__ == '__main__':

    root_dir = '/data/FX/CharImage/'
    font_dir = [ 'CATIA', 'FX_D', 'FX_W1', 'FXSVL', 'ProE' ] 
    #PATHS = (['/data/FX/CharImage/'], '') 
    PATHS = ( [ root_dir + font for font in font_dir], None)

    PATHS = ( ['D:/data/img/BreaKHis/BreaKHis_v1/histology_slides/breast'], None)
    PATHS = ( ['D:\data\img\KaggleBreastHistology'], None)

    NEED_BUILD_TFRECORD = True
    BATCH_SIZE = 20
    K = 10
    #TEST_IDXES = list(range(0,K-3))
    TEST_IDXES = [9]

    #d = HandleIIDDataTFRecord( 'CharImages', BATCH_SIZE, K, PATHS, is_debug=False)
    #d = HandleIIDDataTFRecord( 'CIFAR10', BATCH_SIZE, K, PATHS, is_debug=False)
    d = HandleIIDDataTFRecord( 'SVHN', is_debug=False)
    #d = HandleIIDDataTFRecord( 'BreaKHis', BATCH_SIZE, K, PATHS, is_debug=False)
    #d = HandleIIDDataTFRecord( 'KaggleBreastHistology', BATCH_SIZE, K, PATHS, is_debug=False)

    if NEED_BUILD_TFRECORD:
        o = d.prepare( do_write_tfrecord = True,
                       do_write_npy      = False)

    # get as tf.QUEUE
    with tf.Graph().as_default() as g:
        print(d.get_tfrecords(TEST_IDXES))

    # get as numpy
    d.get_ndarrays(TEST_IDXES)

    sys.exit('saigo')
