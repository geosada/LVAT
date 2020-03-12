import tensorflow as tf
import numpy as np
import os,sys

# to import models
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../CNN')


"""
#tf.flags.DEFINE_string("data_set", "CIFAR10", "SVHN /CIFAR10 / CelebA")
tf.flags.DEFINE_string("data_set", "CelebA", "SVHN /CIFAR10 / CelebA")
tf.flags.DEFINE_boolean("restore", False, "restore from the last check point")
"""
tf.flags.DEFINE_string("dir_root", "./out/", "")
tf.flags.DEFINE_string("file_ckpt", "", "")

FLAGS = tf.flags.FLAGS

if FLAGS.data_set == "CelebA":
    WIDTH_RESNET  = 128
    N_FLOW_STEPS  = 22
    N_FLOW_STEPS  = 32
    N_FLOW_SCALES = 4
    IMAGE_SIZE = 64
    BATCH_SIZE    = 4      #
    BATCH_SIZE    = 128     # just for interpolation
    BATCH_SIZE    = 16      #

elif FLAGS.data_set == "SVHN" or FLAGS.data_set == 'CIFAR10':
    WIDTH_RESNET  = 128
    N_FLOW_STEPS  = 22
    N_FLOW_SCALES = 3
    IMAGE_SIZE = 32
    BATCH_SIZE    = 128
else:
    raise ValueError

if FLAGS.is_aug:
    dir_logs = os.path.join(FLAGS.dir_root, FLAGS.data_set + '_aug')
else:
    dir_logs = os.path.join(FLAGS.dir_root, FLAGS.data_set)
dir_logs =  os.path.join(dir_logs, "w_%d__step_%d__scale_%d__b_%d"%(WIDTH_RESNET, N_FLOW_STEPS, N_FLOW_SCALES, BATCH_SIZE))

FLAGS.file_ckpt = os.path.join(dir_logs,"model.ckpt")

print('checkpoint:', FLAGS.file_ckpt)
os.makedirs(dir_logs, exist_ok=True)

IS_DRYRUN = False
if IS_DRYRUN:
    sess = tf.InteractiveSession()
    a = tf.Variable(0)
    sess.run(tf.global_variables_initializer())                                                                                                                                   
    saver = tf.train.Saver()
    save_path = saver.save(sess, FLAGS.file_ckpt)
    print("Dryrun ... Model will be saved in path: %s" % save_path)
    sys.exit('exit dry run')

