import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf

import layers_vat as L
import vat
import utils  as u
from collections import namedtuple
import sys, os
import math


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")
tf.app.flags.DEFINE_string('data_set', 'SVHN', "{CIFAR10, SVHN}")
tf.app.flags.DEFINE_string('log__dir', "./out", "log_dir")

tf.app.flags.DEFINE_string('data__dir', '/data/img/SVHN/', "")
tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")
tf.app.flags.DEFINE_integer('num_labeled_examples', 1000, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_valid_examples', 1000, "The number of validation examples")

tf.app.flags.DEFINE_integer('seed', 1, "initial random seed")
tf.app.flags.DEFINE_bool('validation', False, "")

tf.app.flags.DEFINE_integer('batch_size', 32, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('ul_batch_size', 128, "the number of unlabeled examples in a batch")
tf.app.flags.DEFINE_integer('eval_batch_size', 25, "the number of eval examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq', 1, "")
tf.app.flags.DEFINE_integer('eval_start', 150, "")
tf.app.flags.DEFINE_integer('num_epochs', 120, "the number of epochs for training")
tf.app.flags.DEFINE_integer('epoch_decay_start', 80, "epoch of starting learning rate decay")
tf.app.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "initial leanring rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")

tf.app.flags.DEFINE_bool('draw_adv_img',  False, "")
tf.app.flags.DEFINE_string('method', 'lvat', "{vat, vatent, lvat, baseline}")
tf.app.flags.DEFINE_float('alpha', 1.0, "")
tf.app.flags.DEFINE_boolean("is_aug",  False, "data augmentation")
tf.app.flags.DEFINE_bool('is_aug_trans',  False, "")
tf.app.flags.DEFINE_bool('is_aug_flip',   False, "")
tf.app.flags.DEFINE_bool('is_aug_rotate', False, "")
tf.app.flags.DEFINE_bool('is_aug_color',  False, "")
tf.app.flags.DEFINE_bool('is_aug_crop',   False, "")

tf.app.flags.DEFINE_string('ae_type', 'Glow', "{VAE, Glow}")


tf.app.flags.DEFINE_integer('img_size', 1, "image size")
tf.app.flags.DEFINE_integer('n_class',  1, "n of classicification class size")

if FLAGS.data_set == 'CIFAR10':
    from cifar10 import inputs, unlabeled_inputs
    FLAGS.img_size = 32
    FLAGS.n_class  = 10
elif FLAGS.data_set == 'SVHN':
    from svhn import inputs, unlabeled_inputs 
    FLAGS.img_size = 32
    FLAGS.n_class  = 10
else: 
    raise NotImplementedError

NUM_EVAL_EXAMPLES = 5000

SCOPE_CLASSIFIER = 'scope_classifier'
if FLAGS.ae_type == 'VAE':
    SCOPE_ENCODER    = 'scope_vae'
    SCOPE_DECODER    = 'scope_vae'
    if FLAGS.data_set == 'CIFAR10':
        CKPT_AE = 'out_VAE_CIFAR10'
    elif FLAGS.data_set == 'SVHN':
        CKPT_AE = 'out_VAE_SVHN'
    else:
        raise NotImplementedError

elif FLAGS.ae_type == 'Glow':
    SCOPE_ENCODER    = 'scope_glow'
    SCOPE_DECODER    = 'scope_glow'
    if FLAGS.data_set == 'CelebA':
        CKPT_AE = 'out_Glow_CelebA'
    elif FLAGS.data_set == 'CIFAR10':
        CKPT_AE = 'out_Glow_Cifar10'
    elif FLAGS.data_set == 'SVHN':
        CKPT_AE = 'out_Glow_SVHN'
    else:
        raise NotImplementedError

if FLAGS.is_aug:
    CKPT_AE = CKPT_AE + '_aug'
    if FLAGS.data_set == 'CIFAR10':
        FLAGS.is_aug_trans = True
        FLAGS.is_aug_flip  = True
    elif FLAGS.data_set == 'SVHN':
        FLAGS.is_aug_trans = True
        

def build_training_graph(x, y, ul_x, lr, mom):

    logit = vat.forward(x)

    nll_loss = L.ce_loss(logit, y)
    x_reconst = tf.constant(0)
    if FLAGS.method == 'vat':
        ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
        vat_loss, r_adv = vat.virtual_adversarial_loss(ul_x, ul_logit)
        x_adv = ul_x + r_adv
        additional_loss = vat_loss

    elif FLAGS.method == 'vatent':
        ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
        vat_loss, r_adv = vat.virtual_adversarial_loss(ul_x, ul_logit)
        x_adv = ul_x + r_adv
        ent_loss = L.entropy_y_x(ul_logit)
        additional_loss = vat_loss + ent_loss

    elif FLAGS.method == 'lvat':
        ul_logit = vat.forward(ul_x, is_training=True, update_batch_stats=False)
        
        m_ae = get_ae()
        with tf.variable_scope(SCOPE_ENCODER ):
            if FLAGS.ae_type == 'VAE':
                _,z,_ = m_ae.encoder(ul_x, is_train=False)
            elif FLAGS.ae_type == 'AE':
                z = m_ae.encoder(ul_x, is_train=False)
            elif FLAGS.ae_type == 'Glow':
                print('[DEBUG] ... building Glow encoder')
                with tf.variable_scope('encoder' ):
                    y, logdet, z = m_ae.encoder(ul_x)

        decoder = m_ae.decoder
        if FLAGS.ae_type == 'Glow':
            print('[DEBUG] ... building Glow VAT loss function')
            vat_loss, r_adv_y, r_adv_z = vat.virtual_adversarial_loss_glow((y, logdet, z), ul_logit, decoder)

            print('[DEBUG] ... building Glow decoder')
            with tf.variable_scope(SCOPE_DECODER, reuse=tf.AUTO_REUSE):
                #with tf.variable_scope('decoder' ):
                    x_adv     = decoder((y+r_adv_y, logdet, z+r_adv_z))
                    x_reconst = decoder((y,         logdet, z))

        else:
            vat_loss, r_adv = vat.virtual_adversarial_loss(z, ul_logit, decoder)

            with tf.variable_scope(SCOPE_DECODER, reuse=tf.AUTO_REUSE):
                x_adv     = decoder(z + r_adv, False)
                x_reconst = decoder(z, False)

        additional_loss = vat_loss

    elif FLAGS.method == 'baseline':
        additional_loss = 0
    else:
        raise NotImplementedError

    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom) 
    theta_classifier = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=SCOPE_CLASSIFIER)

def get_ae():
    if FLAGS.ae_type == 'Glow':
        sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../glow/')
        from Glow import Glow
        with tf.variable_scope('scope_glow' ):  # it is needed for Actnorm
            m = Glow()
        return m

    else:
        if FLAGS.ae_type == 'VAE':
            from VAE import VAE
            m = VAE
        elif FLAGS.ae_type == 'AE':
            from AE import AE
            m = AE
        elif FLAGS.ae_type == 'DAE':
            from AE import DAE
            m = DAE
                                                        
        """ ugly code just for compatiblity """
        Resource = namedtuple('Resource', ('dh', 'ph'))
        ph, ph['lr'] = dict(), None
        dummy = Resource(dh=None, ph=ph)
        return m(dummy)
    
def build_eval_graph(x, y, ul_x):
    losses = {}
    logit = vat.forward(x, is_training=False, update_batch_stats=False)
    nll_loss = L.ce_loss(logit, y)
    losses['NLL'] = nll_loss
    acc = L.accuracy(logit, y)
    losses['Acc'] = acc
    scope = tf.get_variable_scope()
    scope.reuse_variables()

    results = {}
    if FLAGS.method == 'vat' or FLAGS.method == 'vatent':
        ul_logit = vat.forward(ul_x, is_training=False, update_batch_stats=False)
        vat_loss, r_adv = vat.virtual_adversarial_loss(ul_x, ul_logit, is_training=False)
        losses['VAT_loss'] = vat_loss
        x_adv = ul_x + r_adv
        x_reconst = ul_x    # dummy for compatible
        y_reconst = tf.argmax(ul_logit, 1)       # dummy for compatible

    elif FLAGS.method == 'lvat':
        ul_logit = vat.forward(ul_x, is_training=False, update_batch_stats=False)

        m_ae = get_ae()
        decoder = m_ae.decoder
        if FLAGS.ae_type == 'Glow':
            print('[DEBUG] ... building Glow encoder in eval graph')
            with tf.variable_scope(SCOPE_ENCODER, reuse=tf.AUTO_REUSE ):
                with tf.variable_scope('encoder' ):
                    y_latent, logdet, z = m_ae.encoder(ul_x)
            lvat_loss, r_adv_y, r_adv_z = vat.virtual_adversarial_loss_glow((y_latent, logdet, z), ul_logit, decoder)
            print('[DEBUG] ... building Glow decoder in eval graph')
            with tf.variable_scope(SCOPE_DECODER, reuse=tf.AUTO_REUSE):
                with tf.variable_scope('decoder' ):
                    x_adv     = decoder((y_latent+r_adv_y, logdet, z+r_adv_z))
                    x_reconst = decoder((y_latent        , logdet, z))

        else:
            with tf.variable_scope(SCOPE_ENCODER, reuse=tf.AUTO_REUSE ):
                if FLAGS.ae_type == 'VAE':
                    _,z,_ = m_ae.encoder(ul_x, is_train=False)
                elif FLAGS.ae_type == 'AE':
                    z = m_ae.encoder(ul_x, is_train=False)
            lvat_loss, r_adv = vat.virtual_adversarial_loss(z, ul_logit, decoder)
            with tf.variable_scope(SCOPE_DECODER, reuse=tf.AUTO_REUSE):
                x_adv     = decoder(z + r_adv, False)
                x_reconst = decoder(z, False)

        losses['LVAT_loss'] = lvat_loss

        logit_reconst = vat.forward(x_reconst, is_training=False, update_batch_stats=False)
        y_reconst = tf.argmax(logit_reconst, 1)

    results['x']         = ul_x
    results['x_reconst'] = x_reconst
    results['y_reconst'] = y_reconst

    results['x_adv'] = x_adv
    results['y_pred'] = tf.argmax(logit, 1)
    results['y_true'] = tf.argmax(y, 1)

    x = tf.reshape(x, (-1, FLAGS.img_size*FLAGS.img_size*3))
    x_adv = tf.reshape(x_adv, (-1, FLAGS.img_size*FLAGS.img_size*3))
    x_reconst = tf.reshape(x_reconst, (-1, FLAGS.img_size*FLAGS.img_size*3))
    results['x_diff'] = tf.norm( x - x_reconst, axis=1)
    results['x_diff_adv'] = tf.norm( x - x_adv, axis=1)

    return losses, results


def main(_):
    print(FLAGS.epsilon, FLAGS.top_bn)
    np.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(np.random.randint(1234))
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):

            if FLAGS.data_set == 'CelebA':
                (images, labels), (_,_),(_,_) = d.get_data(batch_size=FLAGS.batch_size,image_size=FLAGS.img_size)

                (images_eval_train, labels_eval_train), (_,_),(images_eval_test, labels_eval_test) = \
                    d.get_data(batch_size=FLAGS.eval_batch_size,image_size=FLAGS.img_size)

                ul_images = images
                ul_images_eval_train = images_eval_train
            else:
                images, labels = inputs(batch_size=FLAGS.batch_size,
                                        train=True,
                                        validation=FLAGS.validation,
                                        shuffle=True)
                ul_images = unlabeled_inputs(batch_size=FLAGS.ul_batch_size,
                                             validation=FLAGS.validation,
                                             shuffle=True)
                                                                                                
                images_eval_train, labels_eval_train = inputs(batch_size=FLAGS.eval_batch_size,
                                                              train=True,
                                                              validation=FLAGS.validation,
                                                              shuffle=True)
                ul_images_eval_train = unlabeled_inputs(batch_size=FLAGS.eval_batch_size,
                                                        validation=FLAGS.validation,
                                                        shuffle=True)
                                                                                                
                images_eval_test, labels_eval_test = inputs(batch_size=FLAGS.eval_batch_size,
                                                            train=False,
                                                            validation=FLAGS.validation,
                                                            shuffle=True)

        lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        mom = tf.placeholder(tf.float32, shape=[], name="momentum")

        loss, train_op, x_adv, x_reconst = build_training_graph(images, labels, ul_images, lr, mom)

        # Build eval graph
        if not FLAGS.draw_adv_img:
            losses_eval_train, _ = build_eval_graph(images_eval_train, labels_eval_train, ul_images_eval_train)
            losses_eval_test, results  = build_eval_graph(images_eval_test, labels_eval_test, images_eval_test)

        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)

        if FLAGS.method == 'lvat':
            print('-------------------------------------------')
            print("... restore the variables from frozen model.")  
            u.restore(sess, SCOPE_ENCODER, CKPT_AE)
            print('-------------------------------------------')
            

        #if FLAGS.draw_adv_img:
        if False:
            print("... restore the variables of the classifier. log__dir:", FLAGS.log__dir)
            ckpt = tf.train.get_checkpoint_state(FLAGS.log__dir)
            if ckpt and ckpt.model_checkpoint_path:
                u.restore(sess, SCOPE_CLASSIFIER, FLAGS.log__dir)
                op_init = u.init_uninitialized_vars(sess)
                sess.run(op_init, feed_dict={lr: FLAGS.learning_rate, mom: FLAGS.mom1})
            else:
                sys.exit('failed to restore')
        else:
            print("... init the variables for the classifier to be trained.")


            classifier_vars = tf.get_collection( tf.GraphKeys.VARIABLES, scope=SCOPE_CLASSIFIER)
            print('classifier_vars:', classifier_vars)
            op_init = tf.variables_initializer(classifier_vars)

            optimizer_vars = tf.get_collection( tf.GraphKeys.VARIABLES, scope='scope_optimizer')
            print('optimizer_vars:', optimizer_vars)
            op_init_optimiser = tf.variables_initializer(optimizer_vars)
            sess.run([op_init, op_init_optimiser], feed_dict={lr: FLAGS.learning_rate, mom: FLAGS.mom1})


        tf.train.start_queue_runners(sess=sess)

        if FLAGS.draw_adv_img:

            print('... skip training')

            _x, _x_adv, _x_reconst = sess.run([ul_images, x_adv, x_reconst])
            _N = 7
            print(math.floor(FLAGS.ul_batch_size // _N))

            for i in range(math.floor(FLAGS.ul_batch_size // _N)):
                draw_x(_x, _x_reconst, _x_adv, n_x=_N, offset=i, show_reconst=(FLAGS.method == 'lvat'),
                        filename='ep_%s_%.2f_%d'%(FLAGS.method, FLAGS.epsilon, i))

            sys.exit('exit draw_adv_img')

        else:

            print('... start training')

            for ep in range(FLAGS.num_epochs):
                if ep < FLAGS.epoch_decay_start:
                    feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1}
                else:
                    decayed_lr = ((FLAGS.num_epochs - ep) / float(
                        FLAGS.num_epochs - FLAGS.epoch_decay_start)) * FLAGS.learning_rate
                    feed_dict = {lr: decayed_lr, mom: FLAGS.mom2}

                sum_loss = 0
                start = time.time()
                for i in tqdm(range(FLAGS.num_iter_per_epoch), leave=False):

                    _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)

                    sum_loss += batch_loss
                end = time.time()
                print("Epoch:", ep, "CE_loss_train:", sum_loss / FLAGS.num_iter_per_epoch, "elapsed_time:", end - start, flush=True)

                if (ep >= FLAGS.eval_start) and ((ep + 1) % FLAGS.eval_freq == 0 or ep + 1 == FLAGS.num_epochs):

                    test(sess, losses_eval_train, ep, "train-")
                    test(sess, losses_eval_test,  ep, "test-")

                if ep % 10 == 0:
                    print("Model saved in file: %s" % saver.save(sess, FLAGS.log__dir + '/model.ckpt'))
    return

def test(sess, losses, ep, prefix):

    act_values_dict = {}
                                                                                                                     
    for key, _ in losses.items():
        act_values_dict[key] = 0
    n_iter_per_epoch = NUM_EVAL_EXAMPLES / FLAGS.eval_batch_size

    for i in tqdm(range(int(n_iter_per_epoch)), leave=False):
        values = losses.values()
        act_values = sess.run(list(values))

        for key, value in zip(act_values_dict.keys(), act_values):
            act_values_dict[key] += value

    for key, value in act_values_dict.items():
        print("Epoch:", ep, prefix + key, value / n_iter_per_epoch, flush=True)
    return


def get_statistics(sess, _r):
    # this functin is specialized for eval datasets

    n_iter_per_epoch = NUM_EVAL_EXAMPLES / FLAGS.eval_batch_size

    r = {}
    for key, _ in _r.items():
        r[key] = []      # init

    for i in range(int(n_iter_per_epoch)):
        y_true, y_pred, y_reconst, x_diff, x_diff_adv = sess.run((_r['y_true'], _r['y_pred'], _r['y_reconst'], _r['x_diff'], _r['x_diff_adv']))
        r['y_true'].extend( y_true)
        r['y_pred'].extend( y_pred)
        r['y_reconst'].extend( y_reconst)
        r['x_diff'].extend( x_diff)
        r['x_diff_adv'].extend( x_diff_adv)


    true_pred = np.equal(r['y_true'], r['y_pred']).astype(np.int)
    true_reconst = np.equal(r['y_true'], r['y_reconst']).astype(np.int)
    o = np.array(( true_pred, true_reconst, r['x_diff'], r['x_diff_adv']))
    o = o.transpose()
    file_name = 'stat__%s__ep_%s_%.2f.csv'%(FLAGS.data_set, FLAGS.method, FLAGS.epsilon)
    print('... save csv:', file_name)
    np.savetxt(file_name , o, delimiter=',')

    return


def draw_x(x, x_reconst, x_adv, y=None, n_x=7, is_diff_show=False, offset=0, show_reconst=True, filename=''):

    """
    y: probability distribution
    """
    import matplotlib
    matplotlib.use('Agg')           # noqa: E402
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    DIR_OUT = 'out'

    def _preprocess(a): return np.squeeze(a[offset*n_x:(offset+1)*n_x]*255).astype(np.uint8)
    x         = _preprocess(x)
    x_adv     = _preprocess(x_adv)
    if show_reconst:
        x_reconst = _preprocess(x_reconst)

    if len(x) == 0:
        print('len(x) == 0')
        return

    if show_reconst:
        n_rows = 4 if is_diff_show else 3
    else:
        n_rows = 3 if is_diff_show else 2

    fig = plt.figure(figsize=(n_x+0.2, n_rows), constrained_layout=False)

    gs = gridspec.GridSpec(n_rows, n_x+1, width_ratios=[1]*n_x + [0.0],
                           wspace=0.01, hspace=0.01)
    
    for i in range(n_x):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(x[i], interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
    
        if is_diff_show:
            ax = fig.add_subplot(gs[1, i])
            img = ax.imshow(x_adv[i]-x[i], cmap='RdBu_r', vmin=-1,
                            vmax=1, interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
    
        if show_reconst:
            ax = fig.add_subplot(gs[n_rows - 2, i])
            ax.imshow(x_reconst[i], interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
    
        ax = fig.add_subplot(gs[n_rows - 1, i])
        ax.imshow(x_adv[i], interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
    
        if y is not None:
            ax.set_xlabel('{0:.2f}'.format( y[i]), fontsize=12)
    
    IS_COLORBAR_SHOW = False
    if is_diff_show and IS_COLORBAR_SHOW:
        ax = fig.add_subplot(gs[1, n_x])
        dummy = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-1, vmax=1))
        dummy.set_array([])
        fig.colorbar(mappable=dummy, cax=ax, ticks=[-1, 0, 1], ticklocation='right')
    
    #gs.tight_layout(fig)

    dir_img = DIR_OUT + '/img'
    file_img = '%s/%s_%s.png'%(dir_img, FLAGS.data_set, filename)
    os.makedirs(DIR_OUT , exist_ok=True)
    os.makedirs(dir_img, exist_ok=True)
    plt.savefig(file_img)
    print('finished saving figure:', file_img)

if __name__ == "__main__":
    tf.app.run()
