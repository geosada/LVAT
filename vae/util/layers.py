#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import config as c

eps = 1e-8 # epsilon for numerical stability


class Layers(object):

    def __init__(self):
        self.do_share = False

    def set_do_share(self, flag):
        self.do_share = flag

    def W( self, W_shape,  W_name='W', W_init=None):
        if W_init is None:
            W_initializer = tf.contrib.layers.xavier_initializer()
        else:
            W_initializer = tf.constant_initializer(W_init)

        return tf.get_variable(W_name, W_shape, initializer=W_initializer)

    def Wb( self, W_shape, b_shape, W_name='W', b_name='b', W_init=None, b_init=0.1):

        W = self.W(W_shape, W_name=W_name, W_init=None)
        b = tf.get_variable(b_name, b_shape, initializer=tf.constant_initializer(b_init))

        def _summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                """
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                """
                tf.summary.histogram('histogram', var)
        _summaries(W)
        _summaries(b)

        return W, b


    def denseV2( self, scope, x, output_dim, activation=None):
        return tf.contrib.layers.fully_connected( x, output_dim, activation_fn=activation, reuse=self.do_share, scope=scope)

    def dense( self, scope, x, output_dim, activation=None):
        if len(x.get_shape()) == 2:   # 1d
            pass
        elif len(x.get_shape()) == 4: # cnn as NHWC
            #x = tf.reshape(x, [tf.shape(x)[0], -1]) # flatten
            x = tf.reshape(x, [x.get_shape().as_list()[0], -1]) # flatten
            #x = tf.reshape(x, [tf.cast(x.get_shape()[0], tf.int32), -1]) # flatten
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb([x.get_shape()[1], output_dim], [output_dim])
        #with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb([x.get_shape()[1], output_dim], [output_dim])
        o = tf.matmul(x, W) + b 
        return o if activation is None else activation(o)
    
    def lrelu(self, x, a=0.1):
        if a < 1e-16:
            return tf.nn.relu(x)
        else:
            return tf.maximum(x, a * x)

    #def prelu(self, x):

    def multi_conv1d_3(self, seq, filterset):
        """
        seqs: fine grain sequences, the total number of which corresponds to the length of long-term/coarse sequence.
              rank 4. list of (# of fine grain seqs(= len_of_coarse_seq), B, dimx)
        filterset: 1d conv filter_size(= width of filter):the number of filters having
                    e.g.) {3:6, 4:5}
        return: list of coarse seqs
                (len_of_coarse_seq, B, h), where h = sum(filters.values()), total number of filters
        """
        o = [] # coarse_grain_seqs
        #with tf.variable_scope('a_coarse_grain_seq') as vs:
        
        hs = []
        # hs -> list of (B, 1), which lenght is h.
        
        filters = []
        for w,n in filterset.items():
            filters .extend([w]*n)
        # {3:5, 4:5} -> [3,3,3,3,3,4,4,4,4,4]

        for w,n in filterset.items():
            for i in range(n):

                h = self.conv1d('conv1d_'+str(w)+'_'+str(i), seq, 1,
                              filter_size=w, on_time_direction=False)
                # h -> (L, B, 1)
        
                # to strip the rank 3 that has single dimension, 2 is include following arg axis
                h = tf.reduce_max(h, axis=(0, 2))
                # h -> (B,)  hs is batch_size length list of scalar.
                hs.append(h) 
        
        hs = tf.stack(hs)
        # hs -> (h, B)
        hs = tf.transpose(hs)
        # hs -> (B, h)
        o.append(hs)
        #vs.reuse_variables()
        
        return tf.stack(o)

    def multi_conv1d(self, seqs, filterset):
        """
        seqs: fine grain sequences, the total number of which corresponds to the length of long-term/coarse sequence.
              rank 4. list of (# of fine grain seqs(= len_of_coarse_seq), B, dimx)
        filterset: 1d conv filter_size(= width of filter):the number of filters having
                    e.g.) {3:6, 4:5}
        return: list of coarse seqs
                (len_of_coarse_seq, B, h), where h = sum(filters.values()), total number of filters
        """
        # lists of rank 3 tensor [(T_f,B,dim_x)] * T_c -> rank 4 tensor (T_c,T_f,B,dim_x) 
        seqs = tf.stack(seqs)

        def _conv(seq):
            hs = [] # hs -> [(B,)] * h
            for w,n in filterset.items():
                for i in range(n):
                                                                                                     
                    h = self.conv1d('conv1d_'+str(w)+'_'+str(i), seq, 1,
                                  filter_size=w, on_time_direction=False)
                    # h -> (L, B, 1)
            
                    # to strip the rank 3 that has single dimension, 2 is include following arg axis
                    h = tf.reduce_max(h, axis=(0, 2))
                    # h -> (B,)  hs is batch_size length list of scalar.
                    hs.append(h) 
            
            return tf.transpose(tf.stack(hs)) # return as (B, h)

        # ugly workaround to init W and b used in conv1d before execution of tf.map_fn().
        with tf.variable_scope('a_coarse_grain_seq') as vs:
            _conv(seqs[0])
        vs.reuse_variables()
        return tf.map_fn( lambda seq: _conv(seq), seqs)
        

    def multi_conv1d_old(self, seqs, filterset):
        """
        seqs: fine grain sequences, the total number of which corresponds to the length of long-term/coarse sequence.
              rank 4. list of (# of fine grain seqs(= len_of_coarse_seq), B, dimx)
        filterset: 1d conv filter_size(= width of filter):the number of filters having
                    e.g.) {3:6, 4:5}
        return: list of coarse seqs
                (len_of_coarse_seq, B, h), where h = sum(filters.values()), total number of filters
        """
        o = [] # coarse_grain_seqs
        with tf.variable_scope('a_coarse_grain_seq_2') as vs:
            for seq in seqs:
        
                hs = []
                # hs -> list of (B, 1), which lenght is h.
        
                for w,n in filterset.items():
                    for i in range(n):
        
                        h = self.conv1d('conv1d_'+str(w)+'_'+str(i), seq, 1,
                                      filter_size=w, on_time_direction=False)
                        # h -> (L, B, 1)
        
                        # to strip the rank 3 that has single dimension, 2 is include following arg axis
                        h = tf.reduce_max(h, axis=(0, 2))
                        # h -> (B,)  hs is batch_size length list of scalar.
                        hs.append(h) 
                
                hs = tf.stack(hs)
                # hs -> (h, B)
                hs = tf.transpose(hs)
                # hs -> (B, h)
                o.append(hs)
                vs.reuse_variables()
        
        return tf.stack(o)

    def avg_pool(self, x, ksize=2, stride=2):
        return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')
    
    def max_pool_2d(self, x, ksize=2, stride=2):
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

    def max_pool_1d_old_181101(self, x, ksize=2, stride=2):
        return tf.nn.max_pool(x, ksize=[1, 1, ksize, 1], strides=[1, 1, stride, 1], padding='SAME')

    def conv2d( self, scope, x, out_c, filter_size=(3,3), strides=(1,1,1,1), padding="SAME", activation=None):
        """
        x:       [BATCH_SIZE, in_height, in_width, in_channels]
        filter : [filter_height, filter_width, in_channels, out_channels]
        """
        filter = [filter_size[0], filter_size[1], int(x.get_shape()[3]), out_c]
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb(filter, [out_c])
        o = tf.nn.conv2d(x, W, strides, padding) + b
        return o if activation is None else activation(o)

    def conv1d( self, scope, x, out_c, filter_size, stride=1, padding="SAME", 
                is_time_major=True, on_time_direction=True, activation=None, is_pre_act_enable=True):
        """
        x:       [in_width(= seq_length), BATCH_SIZE, in_channels(= dim_x)] if is_time_major=True,
                 otherwise [BATCH_SIZE, in_width(= seq_length), in_channels(= dim_x)] is assumed.
        filter : [filter_height, in_channels, out_channels]
        on_time_direction: if True, conv1d runs on time dimension.
        is_pre_act_enable: see http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        """
        if is_time_major:
            x = tf.transpose(x, perm=[1,0,2])
            # -> [BATCH_SIZE, in_width(= seq_length), in_channels(= dim_x)] 

        if on_time_direction:
            x = tf.transpose(x, perm=[0,2,1])
            # from [BATCH_SIZE, in_width(= seq_length), in_channels(= dim_x)] 
            #   -> [BATCH_SIZE, in_width(= dim_x), in_channels(= seq_length)] 
        filter = [filter_size, int(x.get_shape()[2]), out_c]
        with tf.variable_scope(scope+'conv1d',reuse=self.do_share): W, b = self.Wb(filter, [out_c])

        if is_pre_act_enable:
            if activation is not None: x = activation(x)

        o = tf.nn.conv1d(x, W, stride, padding, data_format="NHWC") + b

        if not is_pre_act_enable:
            if activation is not None: o = activation(o)

        if on_time_direction:
            o = tf.transpose(o, perm=[0,2,1])
            # from [BATCH_SIZE, in_width(= dim_x), in_channels(= seq_length)] 
            #   -> [BATCH_SIZE, in_width(= seq_length), in_channels(= dim_x)] 
        if is_time_major:
            # back to time-major
            o = tf.transpose(o, perm=[1,0,2])
        return o

    def dilated_conv1d( self, scope, x, out_c, filter_size, dilation_rate=[1], padding="SAME", 
                is_time_major=True, on_time_direction=True, activation=None, is_pre_act_enable=True):
        """
        x:       [in_width(= seq_length), BATCH_SIZE, in_channels(= dim_x)] if is_time_major=True,
                 otherwise [BATCH_SIZE, in_width(= seq_length), in_channels(= dim_x)] is assumed.
        filter : [filter_height, in_channels, out_channels]
        on_time_direction: if True, conv1d runs on time dimension.
        is_pre_act_enable: see http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        refer to https://github.com/sjvasquez/web-traffic-forecasting/blob/master/cf/tf_utils.py
        """
        if is_time_major:
            x = tf.transpose(x, perm=[1,0,2])
            # -> [BATCH_SIZE, in_width(= seq_length), in_channels(= dim_x)] 

        if on_time_direction:
            x = tf.transpose(x, perm=[0,2,1])
            # from [BATCH_SIZE, in_width(= seq_length), in_channels(= dim_x)] 
            #   -> [BATCH_SIZE, in_width(= dim_x), in_channels(= seq_length)] 
        filter = [filter_size, int(x.get_shape()[2]), out_c]
        with tf.variable_scope(scope+'dilated_conv1d',reuse=self.do_share): W, b = self.Wb(filter, [out_c])

        if is_pre_act_enable:
            if activation is not None: x = activation(x)

        o = tf.nn.convolution(x, W, padding, dilation_rate=dilation_rate) + b

        if not is_pre_act_enable:
            if activation is not None: o = activation(o)

        if on_time_direction:
            o = tf.transpose(o, perm=[0,2,1])
            # from [BATCH_SIZE, in_width(= dim_x), in_channels(= seq_length)] 
            #   -> [BATCH_SIZE, in_width(= seq_length), in_channels(= dim_x)] 
        if is_time_major:
            # back to time-major
            o = tf.transpose(o, perm=[1,0,2])
        return o

    #def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    def deconv2d_test(self, scope, input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
        filter = [k_h, k_w, output_shape[-1], input_.get_shape()[-1]]
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb(filter, [output_shape[-1]])
            
        deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
    
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
    
        return deconv

    def deconv2d( self, scope, x, out_c, output_size=None, filter_size=(4,4), strides=(1,2,2,1), padding="SAME", activation=None):
        """
        x:  [BATCH_SIZE, in_height, in_width, in_channels]
            output shape is [BATCH_SIZE, in_height*2, in_width*2, out_c] as the way of DCGAN.
            default values are aslo set as DCGAN https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py """
        _b, in_h, in_w, in_c = map( lambda _x: int(_x), x.get_shape())
        filter = [filter_size[0], filter_size[1], out_c, in_c]
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb(filter, [out_c])

        if output_size is None:
            output_shape = ( _b, in_h*2, in_w*2, out_c )
        else:
            print('output_size:', output_size )
            output_shape = ( _b, output_size[0], output_size[1], out_c )
            
        o = tf.nn.conv2d_transpose( x, W, output_shape=output_shape, strides=strides) + b
        return o if activation is None else activation(o)
            
    def resblock( self, scope, input, is_train, filter_size=(3,3), strides=(1,1,1,1), padding="SAME", activation=tf.nn.elu):
        x = input
        if len(x.get_shape()) == 2:   # 1d
            out_c = x.get_shape()[1]
        elif len(x.get_shape()) == 4: # cnn as NHWC
            out_c = x.get_shape()[3]
        _prefix = scope
        scope = _prefix + '_1'

        x = self.bn(scope, x, is_train)
        x = activation(x)
        x = self.conv2d( scope, x, out_c)

        scope = _prefix + '_2'
        x = self.bn(scope, x, is_train)
        x = activation(x)
        #x = tf.cond(is_train, lambda:x, lambda:tf.nn.dropout(x, keep_prob=0.9))
        x = self.conv2d( scope, x, out_c)
        return input + x

    def highway( self, scope, x, size=None, activation=tf.nn.elu, carry_bias=-2.0):

        """
        x: (B, dim_x)
        """
        if size is None: size = x.get_shape()[1]
        with tf.variable_scope(scope,reuse=self.do_share):
            W, b = self.Wb([size, size], [size], W_init=0.1, b_init=carry_bias)
        with tf.variable_scope(scope+'hw_trans_wb', reuse=self.do_share):
            W_T, b_T = self.Wb([size, size], [size], W_init=0.1, b_init=carry_bias)

        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
        H = activation(tf.matmul(x, W) + b, name="activation")
        C = tf.subtract(1.0, T, name="carry_gate")

        o = tf.add(tf.multiply(H, T), tf.multiply(x, C), 'y') # y = (H * T) + (x * C)
        return o

    def highway_conv2d( self, scope, x, out_c, filter_size=(3,3), strides=(1,1,1,1), padding="SAME", carry_bias=-1.0, activation=None):
        """
        https://github.com/fomorians/highway-cnn/blob/master/main.py
        """
        filter = [filter_size[0], filter_size[1], int(x.get_shape()[3]), out_c]
        with tf.variable_scope(scope,reuse=self.do_share): W, b = self.Wb(filter, [out_c], carry_bias)
        with tf.variable_scope(scope+'hw_trans_wb', reuse=self.do_share): W_T, b_T = self.Wb(filter, [out_c])
        H = tf.nn.elu(tf.nn.conv2d(x, W, strides, padding) + b, name='activation')
        T = tf.sigmoid(tf.nn.conv2d(x, W_T, strides, padding) + b_T, name='transform_gate')
        C = tf.subtract(1.0, T, name="carry_gate")
        if x.get_shape()[3] != out_c:
            #print('DDDD')
            h,w,c = int(x.get_shape()[1]), int(x.get_shape()[2]), int(x.get_shape()[3])
            x = tf.reshape(x, [-1, h*w*c])
            #x = tf.reshape(x, [-1, img_size*2])
            x = self.dense( scope+'affine',  x, h*w*out_c)
            x = tf.reshape(x, [-1, h,w,out_c])
        else:
            print('skip affine transform')
        print(x.get_shape())
        o = tf.add(tf.multiply(H, T), tf.multiply(x, C), 'y') # y = (H * T) + (x * C)
        return o if activation is None else activation(o)
    
        
    ###########################################
    """         Region Classifier           """
    ###########################################
    def generate_orthogonals_well( self, x, _n, _e):
        """
        _n: number of samples to be generated
        _e: epsilon. magnitude of perturbation
        return: (_n, original_dim[0], original_dim[1],...)
        """
        _,h,w,c = x.get_shape().as_list()   # b is None
        x = tf.reshape(x, (-1, h*w*c))

        randmat = tf.random_normal((h*w*c, _n))
        q, r = tf.linalg.qr(randmat)
        assert(q.shape == (h*w*c, _n))
        direction = tf.transpose(q) # <- ( _n, h*w*c)

        #norm = tf.norm(dirs, axis=1)

        distance = tf.random_uniform( (_n,), minval=-_e, maxval=_e, dtype=tf.float32)
        perturbation = direction * distance[:,None]

        print(perturbation)

        xs = x[None] + perturbation[:,None,:]
        #xs = x[None] + direction[:,None,:] * distance
        xs = tf.reshape(xs, (-1, _n, h, w, c))

        return xs

        
    def generate_particles_old( self, x, _n, _e, is_orthogonality_enable, _min=None, _max=None):
        """
        _n: number of samples to be generated
        _e: epsilon. magnitude of perturbation
        _min, _max: input space constraint, such as 0 to 255 for images.
        return: (_n, original_dim[0], original_dim[1],...)
        """
        _,h,w,c = x.get_shape().as_list()   # b is None
        x = tf.reshape(x, (-1, h*w*c))

        #
        # generate perturbation
        #
        if is_orthogonality_enable:
            randmat = tf.random_normal((h*w*c, _n))
            q, r = tf.linalg.qr(randmat)
            assert(q.shape == (h*w*c, _n))
            direction = tf.transpose(q) # <- ( _n, h*w*c)
            #norm = tf.norm(dirs, axis=1)
            magnitude = tf.random_uniform( (_n,), minval=-_e, maxval=_e, dtype=tf.float32)
            perturbation = direction * magnitude[:,None]
        else:
            # random direction
            #perturbation = tf.random_uniform((_n, h*w*c))
            #
            # original norm is about 4.8 when _e=0,3
            # like debug: [4.8781757 4.89175   4.9237514 4.743628  4.8414845]
            #perturbation = tf.random_uniform( shape, minval=-_e, maxval=_e, dtype=tf.float32)
            #       perturbation = tf.random_uniform((_n, h*w*c))
            #
            #denom = tf.norm(perturbation, axis=1)
            #perturbation = tf.random_uniform((_n, h*w*c))
            perturbation = tf.random_uniform( (_n, h*w*c), minval=-_e, maxval=_e, dtype=tf.float32)

        #perturbation = tf.nn.l2_normalize(perturbation, 1)
        norm = tf.norm(perturbation, axis=1)

        xs = x[None] + perturbation[:,None,:]
        xs = tf.reshape(xs, (-1, _n, h, w, c))

        # [ToDo] since batch_size is set as None in placeholder, clip_by_value causes error.
        #if _min is not None or _max is not None: 
        #    xs = tf.clip_by_value(xs, _min, _max)
        
        return xs, norm
        #return xs, (norm, magnitude)
    def generate_particles( self, x, _n, _b, _r, is_orthogonality_enable, _min=None, _max=None):
        """
        _n: number of examples to be generated
        _b: batch_size
        _r: radius. maximum distance from original x, magnitude of perturbation
        _min, _max: input space constraint, such as 0 to 255 for images.
        return: (_n, original_dim[0], original_dim[1],...)
        """
        _,h,w,c = x.get_shape().as_list()   # b is None
        x = tf.reshape(x, (-1, h*w*c))

        #
        # generate perturbation
        #
        if is_orthogonality_enable:
            randmat = tf.random_normal((h*w*c, _n))
            q, r = tf.linalg.qr(randmat)
            assert(q.shape == (h*w*c, _n))
            direction = tf.transpose(q) # <- ( _n, h*w*c)
            #norm = tf.norm(dirs, axis=1)
        else:
            # random direction
            #perturbation = tf.random_uniform((_n, h*w*c))
            #
            # original norm is about 4.8 when _r=0,3
            # like debug: [4.8781757 4.89175   4.9237514 4.743628  4.8414845]
            #perturbation = tf.random_uniform( shape, minval=-_r, maxval=_r, dtype=tf.float32)
            #       perturbation = tf.random_uniform((_n, h*w*c))
            #
            #denom = tf.norm(perturbation, axis=1)
            #perturbation = tf.random_uniform((_n, h*w*c))
            #perturbation = tf.random_uniform( (_n, h*w*c), minval=-_r, maxval=_r, dtype=tf.float32)
            direction = tf.random_normal((_n, h*w*c))
            direction = tf.nn.l2_normalize(direction, 1)

        magnitude = tf.random_uniform( (_n,), minval=-1, maxval=1, dtype=tf.float32) * _r
        perturbation = direction * magnitude[:,None]
        #perturbation = tf.nn.l2_normalize(perturbation, 1)
        norm = tf.norm(perturbation, axis=1)        # <- (_n, )
        norm = tf.tile(norm, (_b,))    # <- (b*_n, )

        xs = x[None] + perturbation[:,None,:]   # <- (_b, _n, h*w*c)
        xs = tf.reshape(xs, (-1, _n, h, w, c))

        # [ToDo] since batch_size is set as None in placeholder, clip_by_value causes error.
        #if _min is not None or _max is not None: 
        #    xs = tf.clip_by_value(xs, _min, _max)
        
        return xs, norm

    ###########################################
    """             Softmax                 """
    ###########################################
    def softmax( self, scope, input, size):
        if input.get_shape()[1] != size:
            print("softmax w/ fc:", input.get_shape()[1], '->', size)
            return self.dense(scope, input, size, tf.nn.softmax)
        else:
            print("softmax w/o fc")
            return tf.nn.softmax(input)
    
    ###########################################
    """     Additive Margin Softmax         """
    ###########################################
    def am_softmax_logit(self, x, n_class, is_train, y=None):
        
        '''
        this implementation is highly inspired from Joker316701882's code.
        x : output from the last layer, (batch_size, n_class)
        y : ground truth one-hot label of current training batch
        n_class: number of classes
        '''
        """
        m = 0.1
        m = 0.35    # for MNIST
        s = 30
        s = 1
        """
    
        with tf.name_scope('am_softmax'):
            x = tf.nn.l2_normalize(x, 1, 1e-10)
            w = tf.get_variable(name='w_am_softmax', dtype=tf.float32,
                        shape=[x.get_shape()[1], n_class],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            w = tf.nn.l2_normalize(w, 0, 1e-10)
            cos_theta = tf.matmul(x, w)
            if is_train: 
                #cos_theta = tf.clip_by_value(cos_theta, -1,1) # for numerical steady
                phi = cos_theta - c.SOFTMAX_DEDUCTON 
                o = tf.where(tf.equal(y,1), phi, cos_theta)
            else:
                o = cos_theta
            return c.SOFTMAX_INVERSE_TEMPERATURE*o

    ###########################################
    """          Gumbel Softmax             """
    ###########################################

    """ https://github.com/ericjang/gumbel-softmax """
    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
        """
        def sample_gumbel(shape, eps=1e-20):
            """Sample from Gumbel(0, 1)"""
            U = tf.random_uniform(shape,minval=0,maxval=1)
            return -tf.log(-tf.log(U + eps) + eps)
    
        def gumbel_softmax_sample(logits, temperature):
            """ Draw a sample from the Gumbel-Softmax distribution"""
            y = logits + sample_gumbel(tf.shape(logits))
            return tf.nn.softmax( y / temperature)
    
        y = gumbel_softmax_sample(logits, temperature)
        if hard:
            k = tf.shape(logits)[-1]
            #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y




    ## SAMPLER (VARIATIONAL AUTOENCODER) ##
    
    ###########################################
    """             Split                   """
    ###########################################
    # https://github.com/openai/iaf/blob/master/tf_utils/
    def split(self, x, split_dim, split_sizes):
        #   split_dim:   output dimension, e.g. 1
        #   split_sizes: list of output's elements length, e.g. [30, 30] for mu and siguma to make 30 dim z
        n = len(list(x.get_shape()))
        assert int(x.get_shape()[split_dim]) == np.sum(split_sizes)
        ids = np.cumsum([0] + split_sizes)
        ids[-1] = -1
        begin_ids = ids[:-1]

        ret = []
        for i in range(len(split_sizes)):
            cur_begin = np.zeros([n], dtype=np.int32)
            cur_begin[split_dim] = begin_ids[i]
            cur_end = np.zeros([n], dtype=np.int32) - 1
            cur_end[split_dim] = split_sizes[i]
            ret += [tf.slice(x, cur_begin, cur_end)]
        return ret 

    ###########################################
    """      Rparameterization Tricks       """
    ###########################################
    def epsilon( self, _shape, _stddev=1.):
        return tf.truncated_normal(_shape, mean=0, stddev=_stddev)

    def sampler( self, mu, sigma):
        """
        mu,sigma : (BATCH_SIZE, z_size)
        """
        #return mu + tf.sqrt(sigma)*self.epsilon( tf.shape(mu) )
        return mu + sigma*self.epsilon( tf.shape(mu) )
        
    def vae_sampler( self, scope, x, size, activation=tf.nn.elu):
        # for LVAE
        with tf.variable_scope(scope,reuse=self.do_share): 
            # 171120
            mu       = self.dense(scope+'_vae_mu', x, size, activation)
            #mu       = self.dense(scope+'_vae_mu', x, size)
            logsigma = self.dense(scope+'_vae_logsigma', x, size, activation)
            logsigma = tf.clip_by_value(logsigma, eps, 50)
        sigma = tf.exp(logsigma)
        return self.sampler(mu, sigma), mu, logsigma 

    def vae_sampler_w_feature_slice( self, x, size):
        mu, logsigma = self.split( x, 1, [size]*2)
        logsigma = tf.clip_by_value(logsigma, eps, 50)
        sigma = tf.exp(logsigma)
        return self.sampler(mu, sigma), mu, logsigma 

    def precision_weighted( self, musigma1, musigma2):
        mu1, sigma1 = musigma1
        mu2, sigma2 = musigma2
        sigma1__2 = 1 / tf.square(sigma1)
        sigma2__2 = 1 / tf.square(sigma2)
        mu = ( mu1*sigma1__2 + mu2*sigma2__2 )/(sigma1__2 + sigma2__2)
        sigma = 1 / (sigma1__2 + sigma2__2)
        logsigma = tf.log(sigma + eps)
        return (mu, logsigma, sigma)
    
    def precision_weighted_sampler( self, scope, musigma1, musigma2):
        # assume input Tensors are (BATCH_SIZE, dime)
        mu1, sigma1 = musigma1
        mu2, sigma2 = musigma2
        size_1 = mu1.get_shape().as_list()[1]
        size_2 = mu2.get_shape().as_list()[1]

        if size_1 > size_2:
            print('convert 1d to 1d:', size_2, '->', size_1)
            with tf.variable_scope(scope,reuse=self.do_share): 
                mu2       = self.dense(scope+'_lvae_mu', mu2, size_1)
                sigma2 = self.dense(scope+'_lvae_logsigma', sigma2, size_1)
                musigma2  = (mu2, sigma2)
        elif size_1 < size_2:
            raise ValueError("musigma1 must be equal or bigger than musigma2.")
        else:
            # not need to convert
            pass

        mu, logsigma, sigma = self.precision_weighted( musigma1, musigma2)
        return (mu + sigma*self.epsilon(tf.shape(mu) ), mu, logsigma)
    
    ###########################################
    """        Noise/Denose Function        """
    ###########################################
    def get_corrupted(self, x, noise_std=.10):
        return self.sampler( x, noise_std)

    def g_gauss( self, h_c, u):
        """ https://github.com/RobRomijnders/ladder
        element-wise gaussian denoising function proposed in the original paper
        z_c: corrupted latent variable
        u: Tensor from layer (l+1) in decorder
        #size: number hidden neurons for this layer """
        #wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)

        shape = h_c.get_shape().as_list()
        if len(shape) == 2:   # 1d
            W_shape = (shape[1])
        elif len(shape) == 4: # cnn as NHWC
            W_shape = (shape[1], shape[2], shape[3])
        
        def wi( W_init, W_name):
                W = self.W(W_shape, W_name, W_init) 
                return W

        shape_u = u.get_shape().as_list()
        scope = 'g_gause'
        if shape != shape_u:
            if len(shape_u) == 2 and len(shape) == 2 :   # 1d to 1d
                print('convert 1d to 1d:', shape_u, '->', shape)
                u = self.dense(scope, u, shape[1])
            elif len(shape_u) == 2 and len(shape) == 4 :   # 1d to 3d
                print('convert 1d to 3d:', shape_u, '->', shape)
                u = tf.reshape( u, (shape_u[0], shape[1], shape[2], -1))    # reshape based on image size of h_c, so
                u = self.conv2d( scope, u, shape[3])
            elif len(shape_u) == 4 and len(shape) == 4 :    # 3d to 3d
                print('convert 3d to 3d:', shape_u, '->', shape)
                if shape[1]*shape[2]*shape[3] % shape_u[1]*shape_u[2]*shape_u[3] == 0: 
                    u = tf.reshape( u, (shape_u[0], shape[1], shape[2], -1))    # reshape based on image size of h_c, so
                    u = self.conv2d( scope, u, shape[3])
                else:
                    sys.exit('Not implemented', shape, shape_u)

            elif len(shape_u) == 4 and len(shape) == 2 :    # 3d to 1d
                sys.exit('Not implemented')

        with tf.variable_scope(scope,reuse=self.do_share):
            a1 = wi(0., 'a1')
            a2 = wi(1., 'a2')
            a3 = wi(0., 'a3')
            a4 = wi(0., 'a4')
            a5 = wi(0., 'a5')
            a6 = wi(0., 'a6')
            a7 = wi(1., 'a7')
            a8 = wi(0., 'a8')
            a9 = wi(0., 'a9')
            a10 = wi(0., 'a10')
        #Crazy transformation of the prior (mu) and convex-combi weight (v)
        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5   #prior
        v  = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10  #convex-combi weight
        
        h_hat = (h_c - mu) * v + mu  #equation [2] in http://arxiv.org/pdf/1507.02672v2.pdf
        return h_hat
   
    
    def bn_2(self, scope, x, is_train, decay=0.999):
        '''
        https://github.com/OlavHN/bnlstm/blob/master/lstm.py
        '''
    
        if len(x.get_shape()) == 2:   # fc
            size = x.get_shape().as_list()[1]
            axes = [0]
        elif len(x.get_shape()) == 4: # cnn as NHWC
            size = x.get_shape().as_list()[3]
            axes = [0,1,2]
        batch_mean, batch_var = tf.nn.moments(x, axes)
        with tf.variable_scope(scope, reuse=self.do_share):
    
            beta = tf.get_variable(name='beta', shape=[size], initializer=tf.constant_initializer(0.1))
            gamma = tf.get_variable(name='gamma', shape=[size])
    
            pop_mean = tf.get_variable(name='pop_mean', shape=[size], initializer=tf.constant_initializer(0.0), trainable=False)
            pop_var = tf.get_variable(name='pop_var', shape=[size], initializer=tf.constant_initializer(1.0), trainable=False)
    
            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
    
            def batch_statistics():
                with tf.control_dependencies([train_mean_op, train_var_op]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, gamma, beta, eps)
    
            def population_statistics():
                return tf.nn.batch_normalization(x, pop_mean, pop_var, gamma, beta, eps)
    
            return tf.cond(is_train, batch_statistics, population_statistics)

    def bn_1(self, scope, x, is_train=True, do_update_bn=True, collections=None, name="bn", decay=0.999):
    
        if len(x.get_shape()) == 2:   # fc
            size = x.get_shape().as_list()[1]
            axes = [0]
        elif len(x.get_shape()) == 4: # cnn as NHWC
            size = x.get_shape().as_list()[3]
            axes = [0,1,2]
        #params_shape = (dim,)
        with tf.variable_scope(scope, reuse=self.do_share):
            n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
            axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
            mean = tf.reduce_mean(x, axis)
            var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
            avg_mean = tf.get_variable(
                name=name + "_mean",
                #shape=params_shape,
                shape=(size),
                initializer=tf.constant_initializer(0.0),
                collections=collections,
                trainable=False
            )
    
            avg_var = tf.get_variable(
                name=name + "_var",
                #shape=params_shape,
                shape=(size),
                initializer=tf.constant_initializer(1.0),
                collections=collections,
                trainable=False
            )
    
            gamma = tf.get_variable(
                name=name + "_gamma",
                #shape=params_shape,
                shape=(size),
                initializer=tf.constant_initializer(1.0),
                collections=collections
            )
    
            beta = tf.get_variable(
                name=name + "_beta",
                #shape=params_shape,
                shape=(size),
                initializer=tf.constant_initializer(0.0),
                collections=collections,
            )
    
            if is_train:
                avg_mean_assign_op = tf.no_op()
                avg_var_assign_op = tf.no_op()
                if do_update_bn:
                    avg_mean_assign_op = tf.assign(
                        avg_mean,
                        decay * avg_mean + (1 - decay) * mean)
                    avg_var_assign_op = tf.assign(
                        avg_var,
                        decay * avg_var + (n / (n - 1)) * (1 - decay) * var)
    
                with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
                    z = (x - mean) / tf.sqrt(1e-6 + var)
            else:
                z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)
    
            return gamma * z + beta


    def bn(self, scope, x, is_train=True, do_update_bn=True, collections=None, name="bn", decay=0.999):
    
        if len(x.get_shape()) == 2:   # fc
            size = x.get_shape().as_list()[1]
            axes = [0]
        elif len(x.get_shape()) == 4: # cnn as NHWC
            size = x.get_shape().as_list()[3]
            axes = [0,1,2]
        #params_shape = (dim,)
        n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
        axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
        mean = tf.reduce_mean(x, axis)
        var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
        avg_mean = tf.get_variable(
            name=name + "_mean",
            #shape=params_shape,
            shape=(size),
            initializer=tf.constant_initializer(0.0),
            collections=collections,
            trainable=False
        )

        avg_var = tf.get_variable(
            name=name + "_var",
            #shape=params_shape,
            shape=(size),
            initializer=tf.constant_initializer(1.0),
            collections=collections,
            trainable=False
        )

        gamma = tf.get_variable(
            name=name + "_gamma",
            #shape=params_shape,
            shape=(size),
            initializer=tf.constant_initializer(1.0),
            collections=collections
        )

        beta = tf.get_variable(
            name=name + "_beta",
            #shape=params_shape,
            shape=(size),
            initializer=tf.constant_initializer(0.0),
            collections=collections,
        )

        if is_train:
            avg_mean_assign_op = tf.no_op()
            avg_var_assign_op = tf.no_op()
            if do_update_bn:
                avg_mean_assign_op = tf.assign(
                    avg_mean,
                    decay * avg_mean + (1 - decay) * mean)
                avg_var_assign_op = tf.assign(
                    avg_var,
                    decay * avg_var + (n / (n - 1)) * (1 - decay) * var)

            with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
                z = (x - mean) / tf.sqrt(1e-6 + var)
        else:
            z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)

        return gamma * z + beta


