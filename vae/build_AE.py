import tensorflow as tf
import numpy as np
import sys, os, time
from collections import namedtuple
from tqdm import tqdm

import config as c

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/util/')


np.set_printoptions(threshold=np.inf)


def test(epoch, m, sess, resource):

    if c.IS_TF_QUEUE_AVAILABLE: xy = None

    time_start = time.time()

    Lr = []

    for i in tqdm(range(m.d.n_batches_test), leave=False):

        if not c.IS_TF_QUEUE_AVAILABLE: xy = m.d.get_next_batch(i, False)
        r = fetch_from_test_graph(m, sess, resource, xy)

        Lr.append( r['Lr'])

    Lr = np.mean(Lr, axis=0)

    elapsed_time = time.time() - time_start
    o = ("validation: epoch:%d,                                                  loss: %5f, time:%.3f" % (epoch, Lr, elapsed_time ))
    print(o)

    IS_CHECK_RECONST_IMG = True
    if IS_CHECK_RECONST_IMG:
        from eval import draw_x
        x, x_reconst = r['x'], r['x_reconst']
        if c.BATCH_SIZE > 50: x, x_reconst = x[:50], x_reconst[:50]

        draw_x(x, x_reconst, y=None, filename='debug_in_training_epoch_%d'%(epoch))

    return

def eval(m, sess, resource, xy):

    x,y = xy
    b = c.BATCH_SIZE
    n = len(xy[1])
    n_batches = n // b
    Lr, x_reconst = [],[]

    assert n_batches>0, 'something wrong, check batch_size and n_samples'
    for i in tqdm(range(n_batches), leave=False):
        r = fetch_from_test_graph(m, sess, resource, (x[i*b:(i+1)*b], y[i*b:(i+1)*b]))
        #print(r)
        Lr.append( r['Lr'])
        x_reconst.extend( r['x_reconst'])

    Lr = np.mean(Lr, axis=0)
    o = ("loss: %5f " % (Lr))
    print(o)
    x_reconst = np.array(x_reconst)

    return x_reconst, Lr

def fetch_from_test_graph(m, sess, resource, xy=None):

    if xy is None: feed_dict = None
    else:
        (_x, _y) = xy
        feed_dict = {resource.ph['x_test']:_x , resource.ph['y_test']:_y}
    return sess.run(m.o_test, feed_dict)



def build(ckpt=None, graph=None):
    #with tf.Graph().as_default() as graph_ae:
    
        ###########################################
        """             Load Data               """
        ###########################################

        ph = {}
        if c.IS_TF_QUEUE_AVAILABLE:
            from HandleIIDDataTFRecord import HandleIIDDataTFRecord
            d = HandleIIDDataTFRecord()
            (x_train, y_train), x, (x_test, y_test) = d.get_tfrecords(c.TEST_IDXES)

        else:
            from HandleImageDataNumpy import HandleImageDataNumpy
            d = HandleImageDataNumpy(c.FLAGS.dataset, c.BATCH_SIZE)
    
            ph['x_train'] = x_train = tf.placeholder(tf.float32, shape=[None, d.h, d.w, d.c], name="ph_x_train")
            #ph['x_train'] = x_train = tf.placeholder(tf.float32, shape=[c.BATCH_SIZE, d.h, d.w, d.c], name="ph_x_train")
            ph['y_train'] = y_train = tf.placeholder(tf.float32, shape=[None, d.l],           name="ph_y_train")
            ph['x']       = x       = tf.placeholder(tf.float32, shape=[None, d.h, d.w, d.c], name="ph_x")
            ph['x_test']  = x_test  = tf.placeholder(tf.float32, shape=[None, d.h, d.w, d.c], name="ph_x_test")
            #ph['x_test']  = x_test  = tf.placeholder(tf.float32, shape=[c.BATCH_SIZE, d.h, d.w, d.c], name="ph_x_test")
            ph['y_test']  = y_test  = tf.placeholder(tf.float32, shape=[None, d.l],           name="ph_y_test")


        ###########################################
        """        Build Model Graphs           """
        ###########################################
        ph['lr'] = tf.placeholder(tf.float32, shape=(), name="ph_learning_rate")

        Resource = namedtuple('Resource', ('dh', 'merged', 'saver', 'ph'))
        resource = Resource(dh=d, merged=None, saver=None, ph=ph)

        if c.GENERATOR_IS == 'VAE':
            from VAE import VAE
            m = VAE( resource )
            scope_name = "scope_vae"

        elif c.GENERATOR_IS == 'AE':
            from AE import AE
            m = AE( resource )
            scope_name = "scope_ae"

        elif c.GENERATOR_IS == 'DAE':
            from DAE import DAE
            m = DAE( resource )
            scope_name = "scope_dae"

        else:
            raise ValueError('invalid arg: c.GENERATOR_IS is %s '%(c.GENERATOR_IS )) 

        with tf.variable_scope(scope_name) as scope:
    
            if c.DO_TRAIN :
                print('... now building the graph for training.')
                m.build_graph_train(x_train,y_train) # the third one is a dummy for future
                scope.reuse_variables()

            if c.DO_TEST :
                print('... now building the graph for test.')
                m.build_graph_test(x_test,y_test)
    
    
        ###########################################
        """              Init                   """
        ###########################################
        init_op = tf.global_variables_initializer()
        #for v in tf.all_variables(): print("[DEBUG] %s : %s" % (v.name,v.get_shape()))
    
        saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        sess  = tf.Session(config=config, graph=graph)
    
    
        if c.FLAGS.restore:
            if not ckpt: ckpt = c.FLAGS.file_ckpt
            print("... restore with:", ckpt)
            saver.restore(sess, ckpt)
        else:
            sess.run(init_op)
    
        resource = resource._replace(merged=tf.summary.merge_all(), saver=saver)
        #tf.get_default_graph().finalize()

        return m, sess, resource

def train(m, sess, resource):

    print('... start training')

    _lr, ratio = c.STARTER_LEARNING_RATE, 1.0
    _barrier_depth, barrier_growth = 0.,0.
    for epoch in range(1, c.N_EPOCHS+1):


        loss, Lr, Lz = [],[],[]
        for i in range(m.d.n_batches_train):
    
            if c.IS_TF_QUEUE_AVAILABLE:
                feed_dict = {resource.ph['lr']:_lr,
                            }
            else:
                _x, _y = m.d.get_next_batch(i, True)
                feed_dict = {resource.ph['lr']:_lr, resource.ph['x_train']:_x , resource.ph['y_train']:_y,
                            }
    
            """ do update """
            time_start = time.time()
            #summary, r, op, current_lr = sess.run([resource.merged, m.o_train, m.op, m.lr], feed_dict=feed_dict)
            r, op, current_lr = sess.run([ m.o_train, m.op, m.lr], feed_dict=feed_dict)
            elapsed_time = time.time() - time_start

            loss.append(r['loss'])
            if c.GENERATOR_IS == 'VAE':
                Lr.append(r['Lr'])
                Lz.append(r['Lz'])
    
            #if i == 0:
            #    print('debug:', r['logit'][-1])

            #if i % 5 == 0 and i != 0:
            #    break

            if ~np.isfinite(r['loss']).all():
                print('mu:', r['mu'])
                print('logsigma', r['logsigma'])
                print(" iter:%2d, Lr: %.5f,  Lz: %.5f, time:%.3f" % \
                     (i, np.mean(np.array(r['Lr'])), np.mean(np.array(r['Lz'])), elapsed_time))
                print('mu:', np.mean(np.array(r['mu'])))
                print('logsigma:', np.mean(np.array(r['logsigma'])))
                sys.exit('nan was detected in loss')

            
            #if i % 100 == 0 and i != 0:
            if i % 500 == 0 and i != 0:
                #print('debug:', r['x'])
                #print('debug:', r['x_reconst'])

                # Debug
                """
                import matplotlib.pyplot as plt
                plt.figure(figsize=(20,20))
                for i in range(30):
                    plt.subplot(5,6,i+1)         
                    plt.imshow(r['debug'][i])
                plt.show()
                """

                if c.GENERATOR_IS == 'VAE':
                    print(" iter:%2d, loss: %.5f, Lr: %.5f,  Lz: %.5f, time:%.3f" % \
                     (i, np.mean(np.array(loss)),  np.mean(np.array(Lr)), np.mean(np.array(Lz)), elapsed_time))
                else:
                    print(" iter:%2d, loss: %.5f, time:%.3f" % \
                     (i, np.mean(np.array(loss)), elapsed_time))
    
        print("training:   epoch:%d, loss: %.5f" % \
              ((epoch, np.mean(np.array(loss)), )), flush=True)

        """ test """
        if c.DO_TEST and epoch % 1 == 0:
            test(epoch, m, sess, resource)
    
        """ save """
        #if epoch % 5 == 0:
        if epoch % 1 == 0:
            print("Model saved in file: %s" % resource.saver.save(sess,c.FLAGS.file_ckpt))
    
        """ learning rate decay"""
        if (epoch % c.DECAY_INTERVAL == 0) and (epoch > c.DECAY_AFTER):
            ratio *= c.DECAY_FACTOR
            _lr = c.STARTER_LEARNING_RATE * ratio
            #print('lr decaying is scheduled. epoch:%d, lr:%f <= %f' % ( epoch, _lr, current_lr))

if __name__ == "__main__":
    m, sess, resource = build()
    
    if c.IS_TF_QUEUE_AVAILABLE: tf.train.start_queue_runners(sess=sess)
    if c.DO_TRAIN: train(m, sess, resource)
    print('... now testing')
    test(0, m, sess, resource)
    print('... done.')
    sess.close()
