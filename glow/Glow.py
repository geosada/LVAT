#!/usr/bin/env python
# coding: utf-8


import os, sys


import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm 


import nets
import flow_layers as fl

import config_glow as c
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util/')

import utils

tf.set_random_seed(0)


class Glow():
    def __init__(self):

        nn_template_fn = nets.OpenAITemplate(
            width=c.WIDTH_RESNET
        )
        
        layers, self.actnorm_layers = nets.create_simple_flow(
            num_steps=c.N_FLOW_STEPS, 
            num_scales=c.N_FLOW_SCALES, 
            template_fn=nn_template_fn
        )
        self.model_flow = fl.ChainLayer(layers)
        self.quantize_image_layer = layers[0]
        

    def encoder(self, x):

        #if x is None:
        #    x = self.x_image_ph
        #flow = fl.InputLayer(self.x_image_ph)

        flow = fl.InputLayer(x)
        output_flow = self.model_flow(flow, forward=True)
        
        
        # ## Prepare output tensors
        
        y, logdet, z = output_flow
        return y, logdet, z 
        #self.y, self.logdet, self.z = output_flow
        #return self.y, self.logdet, self.z
    
    
    def decoder(self, flow, is_to_uint8=False):

        """ flow: (y,logdet,z) """

        #if flow is None:
        #    flow = (self.y_ph, self.logdet_pior_tmp, self.z_ph )

        x_reconst, _, _ = self.model_flow(flow, forward=False)
        if is_to_uint8:
            # return [0,255] non-differentiable
            return self.quantize_image_layer.to_uint8(x_reconst)
        else:
            #  return [0,1] differentiable
            return self.quantize_image_layer.defferentiable_quantize(x_reconst)/255.0
    
    def build_graph_train(self, x=None):

        with tf.variable_scope('encoder') as scope:
            y, logdet, z = self.encoder(x)

        self.y, self.logdet, self.z = y, logdet, z
        logdet_pior_tmp = tf.zeros_like(logdet)

    
        #################################################
        """                 Loss                      """
        #################################################
        # 
        # * Here simply the $-logp(x)$
        
        tfd = tf.contrib.distributions
        
        self.beta_ph = tf.placeholder(tf.float32, [])
        
        y_flatten = tf.reshape(y, [c.BATCH_SIZE, -1])
        z_flatten = tf.reshape(z, [c.BATCH_SIZE, -1])
        
        prior_y = tfd.MultivariateNormalDiag(loc=tf.zeros_like(y_flatten), scale_diag=self.beta_ph * tf.ones_like(y_flatten))
        prior_z = tfd.MultivariateNormalDiag(loc=tf.zeros_like(z_flatten), scale_diag=self.beta_ph * tf.ones_like(z_flatten))
        log_prob_y =  prior_y.log_prob(y_flatten)
        log_prob_z =  prior_z.log_prob(z_flatten)
        
        # ### The MLE loss
        
        loss = log_prob_y + log_prob_z + logdet
        loss = - tf.reduce_mean(loss)
        
        
        # ### The L2 regularization loss 
        
        print('... setting up L2 regularziation')
        trainable_variables = tf.trainable_variables() 
        l2_reg = 0.00001 
        l2_loss = l2_reg * tf.add_n([ tf.nn.l2_loss(v) for v in tqdm(trainable_variables, total=len(trainable_variables), leave=False)])
        
        
        # ### Total loss -logp(x) + l2_loss

        loss_per_pixel = loss / c.IMAGE_SIZE / c.IMAGE_SIZE  
        total_loss = l2_loss + loss_per_pixel 

        # it should be moved to main()
        #sess.run(tf.global_variables_initializer())
        
        #################################################
        """               Trainer                    """
        #################################################

        self.lr_ph = tf.placeholder(tf.float32)
        print('... setting up optimizer')
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(total_loss)
        
        
        # it should be moved to main()
        # ## Initialize Actnorms using DDI
        """
        sess.run(tf.global_variables_initializer())
        nets.initialize_actnorms(
            sess,
            feed_dict_fn=lambda: {beta_ph: 1.0},
            actnorm_layers=actnorm_layers,
            num_steps=10,
        )
        """
        
        
        # ## Train model, define metrics and trainer
        
        print('... setting up training metrics')
        self.metrics = utils.Metrics(50, metrics_tensors={"total_loss": total_loss, "loss_per_pixel": loss_per_pixel, "l2_loss": l2_loss})
        self.plot_metrics_hook = utils.PlotMetricsHook(self.metrics, step=1000)
    
        #################################################
        """               Backward Flow               """
        #################################################

        with tf.variable_scope('decoder') as scope:
            self.x_reconst_train = self.decoder((y, logdet, z))
            
            sample_y_flatten = prior_y.sample()
            sample_y = tf.reshape(sample_y_flatten, y.shape.as_list())
            sample_z = tf.reshape(prior_z.sample(), z.shape.as_list())
            sampled_logdet = prior_y.log_prob(sample_y_flatten)
                                                                                     
            with tf.variable_scope(scope, reuse=True):                                                                         
                self.x_sampled_train = self.decoder((sample_y, sampled_logdet, sample_z))
        return

    def build_graph_test(self, x=None):

        with tf.variable_scope('encoder') as scope:
            y, logdet, z = self.encoder(x)

        self.y, self.logdet, self.z = y, logdet, z
        logdet_pior_tmp = tf.zeros_like(logdet)

        self.y_ph = tf.placeholder(tf.float32, y.shape.as_list())
        self.z_ph = tf.placeholder(tf.float32, z.shape.as_list())

        with tf.variable_scope('decoder') as scope:
            self.x_reconst = self.decoder((y, logdet, z))

            with tf.variable_scope(scope, reuse=True):
                self.x_sampled = self.decoder((self.y_ph, logdet_pior_tmp, self.z_ph))

        """ test code
        with tf.variable_scope('encoder') as scope:
            with tf.variable_scope(scope, reuse=True):                                                                         
                y_2, logdet_2, z_2 = self.encoder(x_reconst)

        with tf.variable_scope('decoder') as scope:
            with tf.variable_scope(scope, reuse=True):                                                                         
                x_reconst_2 = self.decoder((y_2, logdet_2, z_2))
                self.x_reconst_2 = self.quantize_image_layer.to_uint8(x_reconst_2)
        """
    
        """
        x = x_reconst_2
        for i in range(1):
            print('reconst:',i)
            with tf.variable_scope('encoder') as scope:
                with tf.variable_scope(scope, reuse=True):
                    y, logdet, z = self.encoder(x)

            with tf.variable_scope('decoder') as scope:
                with tf.variable_scope(scope, reuse=True):
                    x = self.decoder((y, logdet, z))

        self.x_reconst_n = self.quantize_image_layer.to_uint8(x)
        """

        return
