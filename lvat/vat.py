import tensorflow as tf
import numpy
import sys, os

import layers_vat as L
import cnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('epsilon', 1.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('xi', 1e-6, "small constant for finite difference")

SCOPE_CLASSIFIER = 'scope_classifier'


def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    return cnn.logit(x, is_training=is_training,
                     update_batch_stats=update_batch_stats,
                     stochastic=stochastic,
                     seed=seed)


def forward(x, decoder=None, is_training=True, update_batch_stats=True, seed=1234):

    if decoder is not None:
        # when decoder is given, input x is actually z.

        if FLAGS.ae_type == 'Glow':
            SCOPE_DECODER = "scope_glow"

            with tf.variable_scope(SCOPE_DECODER, reuse=tf.AUTO_REUSE):
                x = decoder(x)

        else:
            if FLAGS.ae_type == 'VAE':
                SCOPE_DECODER = "scope_vae"
            elif FLAGS.ae_type == 'AE':
                SCOPE_DECODER = "scope_ae"
            elif FLAGS.ae_type == 'DAE':
                SCOPE_DECODER = "scope_dae"

            with tf.variable_scope(SCOPE_DECODER, reuse=tf.AUTO_REUSE):
                x = decoder(x, is_train=False)

    with tf.variable_scope(SCOPE_CLASSIFIER, reuse=tf.AUTO_REUSE):
        if is_training:
            return logit(x, is_training=True,
                         update_batch_stats=update_batch_stats,
                         stochastic=True, seed=seed)
        else:
            return logit(x, is_training=False,
                         update_batch_stats=update_batch_stats,
                         stochastic=False, seed=seed)

def forward(x, decoder=None, is_training=True, update_batch_stats=True, seed=1234):

    if decoder is not None:
        # when decoder is given, input x is actually z.

        if FLAGS.ae_type == 'Glow':
            # x must be (y,logdet,z)

            SCOPE_DECODER = "scope_glow"

            with tf.variable_scope(SCOPE_DECODER, reuse=tf.AUTO_REUSE):
                x = decoder(x)

        else:
            if FLAGS.ae_type == 'VAE':
                SCOPE_DECODER = "scope_vae"
            elif FLAGS.ae_type == 'AE':
                SCOPE_DECODER = "scope_ae"
            elif FLAGS.ae_type == 'DAE':
                SCOPE_DECODER = "scope_dae"

            with tf.variable_scope(SCOPE_DECODER, reuse=tf.AUTO_REUSE):
                x = decoder(x, is_train=False)

    with tf.variable_scope(SCOPE_CLASSIFIER, reuse=tf.AUTO_REUSE):
        if is_training:
            return logit(x, is_training=True,
                         update_batch_stats=update_batch_stats,
                         stochastic=True, seed=seed)
        else:
            return logit(x, is_training=False,
                         update_batch_stats=update_batch_stats,
                         stochastic=False, seed=seed)


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), list(range(1, len(d.get_shape()))), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), list(range(1, len(d.get_shape()))), keep_dims=True))
    return d


def generate_virtual_adversarial_perturbation_glow(latent, logit, decoder, is_training=True):

    y, logdet, z = latent
    d_y = tf.random_normal(shape=tf.shape(y))
    d_z = tf.random_normal(shape=tf.shape(z))

    for _ in range(FLAGS.num_power_iterations):
        d_y = FLAGS.xi * get_normalized_vector(d_y)
        d_z = FLAGS.xi * get_normalized_vector(d_z)
        logit_p = logit
        logit_m = forward((y+d_y, logdet, z+d_z), decoder, update_batch_stats=False, is_training=is_training)
        dist = L.kl_divergence_with_logit(logit_p, logit_m)
        grad_y = tf.gradients(dist, [d_y], aggregation_method=2)[0]
        grad_z = tf.gradients(dist, [d_z], aggregation_method=2)[0]
        d_y = tf.stop_gradient(grad_y)
        d_z = tf.stop_gradient(grad_z)

    return (FLAGS.epsilon * get_normalized_vector(d_y), FLAGS.epsilon * get_normalized_vector(d_z))

def generate_virtual_adversarial_perturbation(x, logit, decoder=None, is_training=True):

    # when decoder is given, input x is actually z.

    d = tf.random_normal(shape=tf.shape(x))

    for _ in range(FLAGS.num_power_iterations):
        d = FLAGS.xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = forward(x + d, decoder, update_batch_stats=False, is_training=is_training)
        dist = L.kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return FLAGS.epsilon * get_normalized_vector(d)


def virtual_adversarial_loss_glow(latent, logit, decoder, is_training=True, name="vat_loss"):

    y, logdet, z = latent

    r_vadv_y, r_vadv_z = generate_virtual_adversarial_perturbation_glow(latent, logit, decoder, is_training=is_training)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m = forward((y+r_vadv_y, logdet, z+r_vadv_z), decoder, update_batch_stats=False, is_training=is_training)
    loss = L.kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name), r_vadv_y, r_vadv_z 


def virtual_adversarial_loss(x, logit, decoder=None, is_training=True, name="vat_loss"):
    # when decoder is given, input x is actually z.
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, decoder, is_training=is_training)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m = forward(x + r_vadv, decoder, update_batch_stats=False, is_training=is_training)
    loss = L.kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name), r_vadv


def generate_adversarial_perturbation(x, loss):
    grad = tf.gradients(loss, [x], aggregation_method=2)[0]
    grad = tf.stop_gradient(grad)
    return FLAGS.epsilon * get_normalized_vector(grad)


def adversarial_loss(x, y, loss, is_training=True, name="at_loss"):
    r_adv = generate_adversarial_perturbation(x, loss)
    logit = forward(x + r_adv, is_training=is_training, update_batch_stats=False)
    loss = L.ce_loss(logit, y)
    return loss

def pi_loss(logit_t, logit_s, name="pi_loss"):
    logit_t = tf.stop_gradient(logit_t)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logit_t, logit_s))) + FLAGS.epsilon)
    return tf.identity(loss, name=name)

def sampler(mu, logsigma):
    sigma = tf.exp(logsigma)
    epsilon = FLAGS.epsilon * tf.truncated_normal(tf.shape(mu), mean=0, stddev=1.)
    return mu + sigma*epsilon

def distort(x):
    def _distort(a_image):
        """
        bounding_boxes: A Tensor of type float32.
            3-D with shape [batch, N, 4] describing the N bounding boxes associated with the image. 
        Bounding boxes are supplied and returned as [y_min, x_min, y_max, x_max]
        """
        if FLAGS.is_aug_trans:
            a_image = tf.pad(a_image, [[2, 2], [2, 2], [0, 0]])
            a_image = tf.random_crop(a_image, [32,32,3])

        if FLAGS.is_aug_flip:
            a_image = tf.image.random_flip_left_right(a_image)

        if FLAGS.is_aug_rotate:
            from math import pi
            radian = tf.random_uniform(shape=(), minval=0, maxval=360) * pi / 180 
            a_image = tf.contrib.image.rotate(a_image, radian, interpolation='BILINEAR')

        if FLAGS.is_aug_color:
            a_image = tf.image.random_brightness(a_image, max_delta=0.2)
            a_image = tf.image.random_contrast( a_image, lower=0.2, upper=1.8 )
            a_image = tf.image.random_hue(a_image, max_delta=0.2)

        if FLAGS.is_aug_crop:
            # shape: [1, 1, 4]
            bounding_boxes = tf.constant([[[1/10, 1/10, 9/10, 9/10]]], dtype=tf.float32)
                                                                                                         
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                                (32,32,3), bounding_boxes,
                                min_object_covered=(9.8/10.0),
                                aspect_ratio_range=[9.5/10.0, 10.0/9.5])
                                                                                                          
            a_image = tf.slice(a_image, begin, size)
            """ for the purpose of distorting not use tf.image.resize_image_with_crop_or_pad under """
            a_image = tf.image.resize_images(a_image, [32,32])
            """ due to the size of channel returned from tf.image.resize_images is not being given,
                specify it manually. """
            a_image = tf.reshape(a_image, [32,32,3])
        return a_image

    """ process batch times in parallel """
    return tf.map_fn( _distort, x)


