#!/usr/bin/env python
# coding: utf-8

import os, sys

import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt

import nets
import flow_layers as fl

tf.flags.DEFINE_string("data_set", "CIFAR10", "SVHN /CIFAR10")
tf.flags.DEFINE_boolean("restore", False, "restore from the last check point")
tf.flags.DEFINE_boolean("is_aug",  True, "data augmentation")
FLAGS = tf.flags.FLAGS

import config_glow as c
from Glow import Glow
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util/')
import utils

tf.set_random_seed(0)


epoch = 0


def trainer( n_train, lr, m, sess, saver, n_steps=1000, beta=1.0, do_save_model=False, do_save_image=True):

    for i in range(n_train):
        utils.trainer(
            sess, 
            num_steps=n_steps,
            train_op=m.train_op, 
            feed_dict_fn=lambda: {m.lr_ph: lr, m.beta_ph: beta}, 
            metrics=[m.metrics], 
            hooks=[m.plot_metrics_hook]
        )

        global epoch
        epoch += 1

        if do_save_model:
            print("... saving model: %s" % FLAGS.file_ckpt)
            save_path = saver.save(sess, FLAGS.file_ckpt)

        if do_save_image:
            fig = c.dir_logs + '/' + 'fig_glow__%d'%(epoch)
            print("... saving figure: %s" % fig)

            plt.subplot(121)
            plt.imshow(utils.plot_grid(m.x_sampled_train).eval({m.lr_ph: 0.0, m.beta_ph: 0.9}))
            plt.subplot(122)
            plt.imshow(utils.plot_grid(m.x_sampled_train).eval({m.lr_ph: 0.0, m.beta_ph: 1.0}))
            #plt.show()
            plt.savefig(fig)
    
    
def main():

    sess = tf.InteractiveSession()

    if FLAGS.data_set == "SVHN":
        from HandleSVHN import HandleSVHN
        d = HandleSVHN()
        (x,_), (_,_) = d.get_data(batch_size=c.BATCH_SIZE,image_size=c.IMAGE_SIZE)

    elif FLAGS.data_set == "CIFAR10":
        from HandleCIFAR10 import HandleCIFAR10
        d = HandleCIFAR10()
        (x,_), (_,_) = d.get_data(batch_size=c.BATCH_SIZE,image_size=c.IMAGE_SIZE)

    else:
        raise ValueError

    scope_name = 'scope_glow'
    
    with tf.variable_scope(scope_name ) as scope:
        m = Glow()
        m.build_graph_train(x)

    saver = tf.train.Saver()

    if FLAGS.restore:
        print("... restore with:", c.FLAGS.file_ckpt)
        saver.restore(sess, c.FLAGS.file_ckpt) 
    else:
        sess.run(tf.global_variables_initializer())
        nets.initialize_actnorms(
            sess,
            feed_dict_fn=lambda: {m.beta_ph: 1.0},
            actnorm_layers=m.actnorm_layers,
            num_steps=10,
        )
    
    
    sess.run(m.train_op, feed_dict={m.lr_ph: 0.0, m.beta_ph: 1.0})
    
    # ### Train model
    # 
    # * We start from small learning rate (warm-up)
    
    trainer( 1, 0.0001,  m, sess, saver, n_steps=100 )
    
    trainer( 5, 0.0005,  m, sess, saver, n_steps=100 )

    trainer( 5, 0.0001,  m, sess, saver)

    if FLAGS.is_aug: trainer( 5, 0.0001,  m, sess, saver)
    
    trainer( 5, 0.00005, m, sess, saver)

    if FLAGS.is_aug: trainer( 5, 0.00005, m, sess, saver)
    
    trainer( 5, 0.0001,  m, sess, saver)
    
    trainer( 1, 0.0001,  m, sess, saver, n_steps=0, do_save_model=True)

    plot_metrics_hook.run()
    
if __name__ == "__main__":

    main()
