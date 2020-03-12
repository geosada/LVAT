import tensorflow as tf
import numpy as np
import os,sys

IS_TF_QUEUE_AVAILABLE = True

DO_TRAIN       = True
DO_TEST        = True
DO_VIEW        = False
DO_TEST_DEFENCE = False
DO_REGION_CLASSIFY = False

GENERATOR_IS = "AE"
GENERATOR_IS = "VAEGAN"
GENERATOR_IS = "GAN"
GENERATOR_IS = "VAE"

IS_WGAN = False
IS_BN_ENABLED = True

Z_SIZE = 200
Z_SIZE = 128

BATCH_SIZE = 2
BATCH_SIZE = 9
BATCH_SIZE = 256    # for VAE\GAN purpose


if IS_TF_QUEUE_AVAILABLE:
    BATCH_SIZE = 32       # for VAT(SSL)
    BATCH_SIZE_UL = 128   # for VAT(SSL)
    BATCH_SIZE_TEST = 100 # for VAT(SSL)

tf.flags.DEFINE_string("dataset", "SVHN", "CIFAR10 / SVHN ")
tf.flags.DEFINE_boolean("restore", False, "restore from the last check point")
tf.flags.DEFINE_string("dir_logs", "./out/", "")
tf.flags.DEFINE_string("file_ckpt", "", "")
tf.flags.DEFINE_boolean("use_pi",               True, "")
tf.flags.DEFINE_boolean("use_vat",              False, "")
tf.flags.DEFINE_boolean("use_fgsm",             True,  "")
tf.flags.DEFINE_boolean("use_virtual_cw",       False, "")
tf.flags.DEFINE_boolean("use_am_softmax",       False, "")
tf.flags.DEFINE_boolean("use_particle_barrier", False, "")

FLAGS = tf.flags.FLAGS


def set_condition(_use_pi, _use_vat, _use_fgsm, _use_virtual_cw, _use_am_softmax, _use_particle_barrier, dir_logs=FLAGS.dir_logs):
    if DO_TRAIN:
        pass
    else:
        #if dir_logs is None:
        #    dir_logs = "out_%s_vat_%d__fgsm_%d__vcw_%d__ams_%d__pb_%d" % \
        #            (FLAGS.dataset, int(_use_pi), int(_use_vat), int(_use_fgsm), int(_use_virtual_cw),
        #             int(_use_am_softmax), int(_use_particle_barrier))
        FLAGS.dir_logs = dir_logs
    FLAGS.file_ckpt = os.path.join(FLAGS.dir_logs,"model.ckpt")
    FLAGS.use_pi               = _use_pi 
    FLAGS.use_vat              = _use_vat
    FLAGS.use_fgsm             = _use_fgsm
    FLAGS.use_virtual_cw       = _use_virtual_cw
    FLAGS.use_am_softmax       = _use_am_softmax 
    FLAGS.use_particle_barrier = _use_particle_barrier
    print('[INFO] dir_logs was set as: ', FLAGS.dir_logs)
    return 

set_condition(FLAGS.use_pi, FLAGS.use_vat, FLAGS.use_fgsm, FLAGS.use_virtual_cw, FLAGS.use_am_softmax, FLAGS.use_particle_barrier)


if not DO_TRAIN and not FLAGS.restore:
    print('[WARN] FLAGS.restore is set to True compulsorily')
    FLAGS.restore = True

K = 10
TEST_IDXES = [9]
PATHS = ( [''], None)
    
N_EPOCHS = 500
N_PLOTS  = 2000

IS_AUGMENTATION_ENABLED = True
IS_AUG_TRANS_TRUE    = True
IS_AUG_FLIP_TRUE     = False
IS_AUG_ROTATE_TRUE   = False
IS_AUG_COLOR_TRUE    = False
IS_AUG_CROP_TRUE     = False
IS_AUG_NOISE_TRUE    = False

# learning rate decay
STARTER_LEARNING_RATE = 1e-3
STARTER_LEARNING_RATE = 2e-4    # DCGAN
#STARTER_LEARNING_RATE = 0.001   # VAT
DECAY_AFTER = 2
#DECAY_AFTER = 80 # VAT
DECAY_INTERVAL = 2
DECAY_FACTOR = 0.97

# Pi
PI_COOL_DOWN_START    = 100
PI_COOL_DOWN_DURATION = 350
LAMBDA_PI_MIN = 1
assert PI_COOL_DOWN_DURATION > 1, 'duration must be longer than 1 epoch since LAMBDA_PI would stay at stating point.'
LAMBDA_PI =  np.linspace(1, LAMBDA_PI_MIN, PI_COOL_DOWN_DURATION)


IS_DO_ENABLE = False

# Region Classifier
N_PARTICLES_FOR_REGION_CLASSIFIER = 128  # number of neighbour points to be generated
REGION_RADIUS = 0.3
IS_GENERATE_PARTICLE_LOGIT_W_DROPOUT = True
if IS_GENERATE_PARTICLE_LOGIT_W_DROPOUT:
    IS_DO_ENABLE = True
    REGION_RADIUS = 0.0

if DO_REGION_CLASSIFY:
    if DO_TRAIN:
        pass
        sys.exit('Are you sure to start training w/ region classify ? If yes, comment out this line.')

    if BATCH_SIZE * N_PARTICLES_FOR_REGION_CLASSIFIER > 500:
        print('BATCH_SIZE / N_PARTICLES_FOR_REGION_CLASSIFIER =', BATCH_SIZE, N_PARTICLES_FOR_REGION_CLASSIFIER)
        sys.exit('Maybe too much.')


# Particle Barrier
IS_ORTHOGONALITY_ENABLE = True
N_PARTICLES_FOR_BARRIER = 5  # number of neighbour points to be generated
IS_SOFT_SHELL_ENABLE = False
BARRIER_MODE = 'supervised'


if FLAGS.dataset == "SVHN":
    STARTER_BARRIER_DEPTH_MAX = 0.01
    BARRIER_ACTIVATES_AFTER = 50
    BARRIER_ACTIVATES_AFTER = 0
    BARRIER_GROWTH_INTERVAL = 5
    BARRIER_GROWTH = 1

elif FLAGS.dataset == "CIFAR10":
    STARTER_BARRIER_DEPTH_MAX = 0.01
    BARRIER_ACTIVATES_AFTER = 0
    BARRIER_GROWTH_INTERVAL = 5
    BARRIER_GROWTH = 1


DIVERGENCE = 'least_square'
#DIVERGENCE = 'kl_forward'
#DIVERGENCE = 'kl_reverse'
#DIVERGENCE = 'js'
#DIVERGENCE = 'mmd'

# SOFTMAX
SOFTMAX_DEDUCTON = 0.35    # for MNIST
#SOFTMAX_INVERSE_TEMPERATURE = 30
SOFTMAX_INVERSE_TEMPERATURE = 1

# VAT
IS_RELAXED_KL_ENABLE = False

# Regarding measuring the Distance to Decision Boundary
DDB_N_DIRECTIONS = 10
DDB_STEP = 0.01
DDB_MAX  = 0.5
FILE_OF_DDB_DIRECTIONS  = 'gxr3_directions_%d.npy'%(DDB_N_DIRECTIONS)

# FGSM 
EPSILON_FGSM = 0.1

# attack
ADV_TARGET_CLASS = 0
CW_CONFIDENCE = 20
#CW_MAX_ITERATIONS = 1000
CW_MAX_ITERATIONS = 10000
N_BINARY_SEARCH = 10
BOUND_BINARY_SEARCH = (10**-6, 1) # 1e-06
#BOUND_BINARY_SEARCH = (0.00001, 1) # 1e-06

DIR_DATA  = './data/%s.confidence_%s/'%(FLAGS.dataset, CW_CONFIDENCE)
X_CLASSIFIED_CORRECTLY = DIR_DATA + 'x_classified_correctly.npy'
Y_CLASSIFIED_CORRECTLY = DIR_DATA + 'y_classified_correctly.npy'
X_ORIGINAL             = DIR_DATA + 'x_original.npy'
Y_ORIGINAL             = DIR_DATA + 'y_original.npy'
X_ADVERSARIAL          = DIR_DATA + 'x_adversarial.npy'
Y_ADVERSARIAL_TARGET   = DIR_DATA + 'y_adversarial_target.npy'
X_TRAIN_CLASSIFIED_CORRECTLY = DIR_DATA + 'x_train_classified_correctly.npy'
Y_TRAIN_CLASSIFIED_CORRECTLY = DIR_DATA + 'y_train_classified_correctly.npy'
X_TRAIN_ORIGINAL             = DIR_DATA + 'x_train_original.npy'
Y_TRAIN_ORIGINAL             = DIR_DATA + 'y_train_original.npy'
X_TRAIN_ADVERSARIAL          = DIR_DATA + 'x_train_adversarial.npy'
Y_TRAIN_ADVERSARIAL_TARGET   = DIR_DATA + 'y_train_adversarial_target.npy'
FILE_MODEL = './nn_robust_attacks/models/%s'%(FLAGS.dataset)

# sanitize
SANITIZER = 'AE' 
SANITIZER = 'PGD_X'
SANITIZER = 'PGD_Z'


LAMBDA_PREDICTION_VARIANCE = 0.1
LAMBDA_RECONSTRUCTION = 1
IS_LOSS_PREDICTION_VARIANCE_ENABLED = False
IS_LOSS_RECONSTRUCTION_ENABLED = True

STARTER_LEARNING_RATE_ADV = 1e-3
STARTER_LEARNING_RATE_ADV = 0.2
STARTER_LEARNING_RATE_ADV_GAN = 0.2

N_EPOCHS_ADV = 1000
N_EPOCHS_ADV = 500
N_EPOCHS_ADV = 300
N_EPOCHS_ADV = 100
DECAY_AFTER_ADV = 300
DECAY_INTERVAL_ADV = 100
DECAY_FACTOR_ADV = 0.97

