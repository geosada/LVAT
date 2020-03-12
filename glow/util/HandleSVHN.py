from __future__ import print_function

import os
import os.path

import scipy.io
import scipy.io.wavfile
from imageio import imread  #   for scipy > 1.3.0 
import tensorflow as tf

from tqdm import tqdm
import numpy  as np

from utils import numpy_array_to_dataset
from svhn import load_svhn, DATA_DIR, NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST

tf.flags.DEFINE_bool("do_write_tfrecord", False, "if True, write tfrecords.")
tf.flags.DEFINE_bool("do_write_npy",      True, "if True, write npy.")

FLAGS = tf.flags.FLAGS

class HandleSVHNData(object):

    def __init__(self):
        self.n_train = NUM_EXAMPLES_TRAIN
        self.n_valid = 0
        self.n_test  = NUM_EXAMPLES_TEST

    def get_ndarray(self):

        (x_train, y_train), (x_test, y_test) = load_svhn(True)
        # y is integer at this moment

        #y_train = np.identity(10)[y_train]
        #y_test  = np.identity(10)[y_test ]

        print(x_train[:2])
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        return (x_train, y_train), (x_test, y_test)
        #sys.exit

    def get_data(self,
            image_size: int = 32,
            batch_size: int = 16,
            #batch_size: int = 4,
            buffer_size: int = 512,
            num_parallel_batches: int = 16, # org
            #num_parallel_batches: int = 1,
    ):
        

        def get_dataset_from_ndarray():
            (x_train, y_train), (x_test, y_test) = self.get_ndarray()

            def _zip(x,y):
                """ x,y are np.array """
                x_dataset = numpy_array_to_dataset(x,
                            buffer_size=buffer_size, batch_size=batch_size, num_parallel_batches=num_parallel_batches)
                                                                                                                       
                y_dataset = numpy_array_to_dataset(y,
                            buffer_size=buffer_size, batch_size=batch_size, num_parallel_batches=num_parallel_batches)
                                                                                                                       
                return tf.data.Dataset.zip((x_dataset, y_dataset))

            return _zip(x_train, y_train), _zip(x_test, y_test)

        def get_dataset_from_tfrecord():
            DATASET_SEED = 1
            dir_rood = DATA_DIR + '/seed' + str(DATASET_SEED) 
            train_tfrecord = dir_rood + '/' + 'labeled_train.tfrecords'
            test_tfrecord  = dir_rood + '/' + 'test.tfrecords'
            print('... set up to read from', dir_rood)
            return tf.data.TFRecordDataset(train_tfrecord), tf.data.TFRecordDataset(test_tfrecord)

        def apply_parser(dataset):
            
            def preprocess(example):
                features = tf.parse_single_example(
                    example,
                    features={
                        "image" : tf.FixedLenFeature([32 * 32 * 3], tf.float32),
                        "label": tf.FixedLenFeature([], tf.int64)
                        }
                )
                                                                                                       
                image = tf.reshape(features["image"], [32, 32, 3])
                label  = features["label"]

                return image, label
                                                                                                       
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    map_func=preprocess,
                    batch_size=batch_size,
                    num_parallel_batches=num_parallel_batches,
                    drop_remainder=True,
                )
            )
            return dataset

        def read(dataset):
                                                                                                       

            if buffer_size > 0:
                dataset = dataset.apply(
                    tf.contrib.data.shuffle_and_repeat(buffer_size=buffer_size, count=-1)
                )

            if IS_FROM_TFRECORDS:
                dataset = apply_parser(dataset)
                

            datayyset = dataset.prefetch(4)
            images, label = dataset.make_one_shot_iterator().get_next()
                                                                                                       
            x = tf.reshape(images, [batch_size, 32, 32, 3])
            #x = tf.image.resize_images(
            #    x, [image_size, image_size], method=0, align_corners=False
            #)
            y = tf.one_hot(tf.cast( label, tf.int32), 10)

            return x, y

        IS_FROM_TFRECORDS = True
        if IS_FROM_TFRECORDS:
            ds_train, ds_test = get_dataset_from_tfrecord()
        else:
            ds_train, ds_test = get_dataset_from_ndarray()

        return read(ds_train), read(ds_test)

    def prepare(self):
    
        df_train, df_valid, df_test = get_train_val_test()

        if FLAGS.do_write_tfrecord:
            self.write_tfrecord(df_train, TFRECORD_TRAIN)
            self.write_tfrecord(df_valid, TFRECORD_VALID)
            self.write_tfrecord(df_test,  TFRECORD_TEST)

        if FLAGS.do_write_npy:
            self.write_npy(df_train, NPY_TRAIN)
            self.write_npy(df_valid, NPY_VALID)
            self.write_npy(df_test,  NPY_TEST)
        print('... exit prepare()')
        return
    
    def read_image( self, file_name):
    
        """ file_name: str. jpg image file name"""
    
        #image = imread(os.path.join(FLAGS.fn_root, file_name))
        image = imread(os.path.join(PATH_TO_RAWIMG, file_name))
        #
        # cropping is performed here.
        # this is the same way as RealNVP and kmkolasinski's Glow implemention 
        #
    
        # original shape [218, 178, 3] is going to be converted into [144, 144, 3] once,
        image = image[40:188, 15:163, :]
        image = image.reshape([148, 148, 3]) 

        # then convert it into (IMG_SIZE, IMG_SIZE)
        import cv2
        image = cv2.resize(image, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        return image
    
    def write_npy(self, df, file_name):

        print('... writing npy file.')
        x, y = [],[]
        for i, row in enumerate(tqdm(df.itertuples(name=None), leave=False, total=len(df))):
    
            row = list(row) # tuple to list 
            img_file_name = row.pop(0)
            
            image = self.read_image(img_file_name)
            x.append(image)
            y.append(row)

        x = np.array(x)
        y = np.array(y)
        print(os.path.join(FLAGS.fn_root, 'x_' + file_name)) 
        print(os.path.join(FLAGS.fn_root, 'y_' + file_name)) 
        np.save(os.path.join(FLAGS.fn_root, 'x_' + file_name) , x)
        np.save(os.path.join(FLAGS.fn_root, 'y_' + file_name) , y)
        return

    def write_tfrecord(self, df, file_name):
    
        def make_example( image, attributes):
            """ image: ndarray whose shape is (144, 144, 3))
                attributes: list of 40 features
            """
            
            image = image.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image" : tf.train.Feature(float_list=tf.train.FloatList(value=image)),
                        "attributes": tf.train.Feature(int64_list=tf.train.Int64List(value=attributes))
                    }
                )
            )
            return example
    
        print('... writing', file_name )
        writer = tf.io.TFRecordWriter(file_name)
    
        for i, row in enumerate(tqdm(df.itertuples(name=None), leave=False, total=len(df))):
    
            row = list(row) # tuple to list 
            img_file_name = row.pop(0)
            
            image = self.read_image(img_file_name)
            ex = make_example(image, row)
            writer.write(ex.SerializeToString())

            #if i > 25: break
        writer.close()
    #sys.exit()


if __name__ == "__main__":


    d = HandleSVHNData()

    #if FLAGS.is_write_mode:
    #d.prepare()

    #print(d.get_ndarray())
    print(d.get_data())

    IS_VISUALLY_CHECKING = False
    ###################################################
    """         visually checking                   """
    ###################################################
    if IS_VISUALLY_CHECKING:
        sys.exit()

    n = 24
    (x_train, y_train), (x_test, y_test) = d.get_data(batch_size=n)
    sess = tf.Session()
    _x,_y = sess.run([x_train, y_train])
    _x = _x*255
    from PIL import Image
    for i in range(n):
        x,y = _x[i].astype(np.uint8), _y[i]
        img = Image.fromarray(x)
        img.save('outfile_%s.jpg'%(i))
        print('Label is:', i, y)
