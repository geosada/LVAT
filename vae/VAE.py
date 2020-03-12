import tensorflow as tf
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util')
from layers import Layers
from losses import LossFunctions
import config as c

#from tflearn.layers.normalization import batch_normalization

class VAE(object):

    def __init__(self, resource):


        """ data and external toolkits """
        self.d  = resource.dh  # dataset manager
        self.ls = Layers()
        self.lf = LossFunctions(self.ls, self.d, self.encoder)

        """ placeholders defined outside"""
        if c.DO_TRAIN:
            self.lr  = resource.ph['lr']


    def encoder(self, h, is_train, y=None):

        if is_train:
            _d = self.d
            #_ = tf.summary.image('image', tf.reshape(h, [-1, _d.h, _d.w, _d.c]), 10)

        scope = 'e_1'
        h = self.ls.conv2d(scope+'_1', h, 128, filter_size=(2,2),  strides=(1,2,2,1), padding="VALID")
        h = tf.layers.batch_normalization(h, training=is_train, name=scope)
        h = tf.nn.relu(h)

        scope = 'e_2'
        h = self.ls.conv2d(scope+'_1', h, 256, filter_size=(2,2),  strides=(1,2,2,1), padding="VALID")
        h = tf.layers.batch_normalization(h, training=is_train, name=scope)
        h = tf.nn.relu(h)

        scope = 'e_3'
        h = self.ls.conv2d(scope+'_1', h, 512, filter_size=(2,2),  strides=(1,2,2,1), padding="VALID")
        h = tf.layers.batch_normalization(h, training=is_train, name=scope)
        #h = tf.nn.relu(h)
        h = tf.nn.tanh(h)

        # -> (b, 4, 4, 512)

        print('h:', h)
        #h = tf.reshape(h, (c.BATCH_SIZE, -1))
        h = tf.reshape(h, (-1, 4*4*512))
        print('h:', h)

        #sys.exit('aa')
        h = self.ls.denseV2('top_of_encoder', h, c.Z_SIZE*2, activation=None)
        print('h:', h)
        return self.ls.vae_sampler_w_feature_slice( h, c.Z_SIZE)

    def decoder(self, h, is_train):

        scope = 'top_of_decoder'
        #h = self.ls.denseV2(scope, h, 128, activation=self.ls.lrelu)
        h = self.ls.denseV2(scope, h, 512, activation=self.ls.lrelu)
        print('h:', scope, h)

        h = tf.reshape(h, (-1, 4,4,32))
        print('h:', scope, h)

        scope = 'd_1'
        h = self.ls.deconv2d(scope+'_1', h, 512, filter_size=(2,2))
        h = tf.layers.batch_normalization(h, training=is_train, name=scope)
        h = tf.nn.relu(h)
        print('h:', scope, h)

        scope = 'd_2'
        h = self.ls.deconv2d(scope+'_2', h, 256, filter_size=(2,2))
        h = tf.layers.batch_normalization(h, training=is_train, name=scope)
        h = tf.nn.relu(h)
        print('h:', scope, h)

        scope = 'd_3'
        h = self.ls.deconv2d(scope+'_3', h, 128, filter_size=(2,2))
        h = tf.layers.batch_normalization(h, training=is_train, name=scope)
        h = tf.nn.relu(h)
        print('h:', scope, h)

        scope = 'd_4'
        h = self.ls.conv2d(scope+'_4', h, 3, filter_size=(1,1),  strides=(1,1,1,1), padding="VALID", activation=tf.nn.sigmoid)
        print('h:', scope, h)

        return h

        
    def build_graph_train(self, x_l, y_l):

        o = dict()  # output
        loss = 0

        if c.IS_AUGMENTATION_ENABLED:
            x_l = distorted = self.distort(x_l)

            if c.IS_AUG_NOISE_TRUE:
                x_l = self.ls.get_corrupted(x_l, 0.15)

        z, mu, logsigma = self.encoder(x_l, is_train=True, y=y_l)

        x_reconst = self.decoder(z, is_train=True)

        """ p(x|z) Reconstruction Loss """
        o['Lr'] = self.lf.get_loss_pxz(x_reconst, x_l, 'Bernoulli')
        o['x_reconst'] = x_reconst
        o['x'] = x_l
        loss += o['Lr']


        """ VAE KL-Divergence Loss """
        LAMBDA_VAE = 0.1
        o['mu'], o['logsigma'] = mu, logsigma
        # work around. [ToDo] make sure the root cause that makes kl loss inf
        #logsigma = tf.clip_by_norm( logsigma, 10)
        o['Lz'] = self.lf.get_loss_vae(c.Z_SIZE, mu,logsigma, _lambda=0.0)
        loss += LAMBDA_VAE * o['Lz']


        """ set losses """
        o['loss'] = loss
        self.o_train = o

        """ set optimizer """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5)
        grads = optimizer.compute_gradients(loss)
        for i,(g,v) in enumerate(grads):
            if g is not None:
                #g = tf.Print(g, [g], "g %s = "%(v))
                grads[i] = (tf.clip_by_norm(g,5),v) # clip gradients
            else:
                print('g is None:', v)
                v = tf.Print(v, [v], "v = ", summarize=10000)


        # update ema in batch_normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.op = optimizer.apply_gradients(grads) # return train_op


    def build_graph_test(self, x_l, y_l):

        o = dict()  # output
        loss = 0


        z, mu, logsigma = self.encoder(x_l, is_train=False, y=y_l)

        x_reconst = self.decoder(mu, is_train=False)
        o['x_reconst'] = x_reconst
        o['x'] = x_l
        #o['Lr'] = self.lf.get_loss_pxz(x_reconst, x_l, 'LeastSquare')
        o['Lr'] = self.lf.get_loss_pxz(x_reconst, x_l, 'Bernoulli')
        #o['Lr'] = self.lf.get_loss_pxz(x_reconst, x_l, 'DiscretizedLogistic')
        #o['Lr'] = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_l, x_reconst))
        loss += o['Lr']


        """ set losses """
        o['loss'] = loss
        self.o_test = o


    def distort(self, x):
    
        """
        maybe helpful http://www.redhub.io/Tensorflow/tensorflow-models/src/master/inception/inception/image_processing.py
        """
        _d = self.d

        def _distort(a_image):
            """
            bounding_boxes: A Tensor of type float32.
                3-D with shape [batch, N, 4] describing the N bounding boxes associated with the image. 
            Bounding boxes are supplied and returned as [y_min, x_min, y_max, x_max]
            """
            if c.IS_AUG_TRANS_TRUE:
                a_image = tf.pad(a_image, [[2, 2], [2, 2], [0, 0]])
                a_image = tf.random_crop(a_image, [_d.h, _d.w, _d.c])

            if c.IS_AUG_FLIP_TRUE:
                a_image = tf.image.random_flip_left_right(a_image)

            if c.IS_AUG_ROTATE_TRUE:
                from math import pi
                radian = tf.random_uniform(shape=(), minval=0, maxval=360) * pi / 180 
                a_image = tf.contrib.image.rotate(a_image, radian, interpolation='BILINEAR')

            if c.IS_AUG_COLOR_TRUE:
                a_image = tf.image.random_brightness(a_image, max_delta=0.2)
                a_image = tf.image.random_contrast( a_image, lower=0.2, upper=1.8 )
                a_image = tf.image.random_hue(a_image, max_delta=0.2)

            if c.IS_AUG_CROP_TRUE:
                # shape: [1, 1, 4]
                bounding_boxes = tf.constant([[[1/10, 1/10, 9/10, 9/10]]], dtype=tf.float32)
                                                                                                             
                begin, size, _ = tf.image.sample_distorted_bounding_box(
                                    (_d.h, _d.w, _d.c), bounding_boxes,
                                    min_object_covered=(9.8/10.0),
                                    aspect_ratio_range=[9.5/10.0, 10.0/9.5])
                                                                                                              
                a_image = tf.slice(a_image, begin, size)
                """ for the purpose of distorting not use tf.image.resize_image_with_crop_or_pad under """
                a_image = tf.image.resize_images(a_image, [_d.h, _d.w])
                """ due to the size of channel returned from tf.image.resize_images is not being given,
                    specify it manually. """
                a_image = tf.reshape(a_image, [_d.h, _d.w, _d.c])
            return a_image

        """ process batch times in parallel """
        return tf.map_fn( _distort, x)
