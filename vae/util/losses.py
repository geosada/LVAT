import tensorflow as tf
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import config as c
#np.set_printoptions(threshold=np.inf)


""" VAT hyper params """
if True:
    XI = 10        # small constant for finite difference
    EP = 1.0         # norm length for (virtual) adversarial training
else:
    # orginal values in https://github.com/takerum/vat_tf/blob/master/vat.py
    XI = 1e-6        # small constant for finite difference
    EP = 8.0         # norm length for (virtual) adversarial training

N_POWER_ITER = 1 # the number of power iterations

CONFIDENCE = 0.2

eps = 1e-8

class LossFunctions(object):

    def __init__(self, layers, dataset, encoder=None):

        self.ls = layers
        self.d  = dataset
        self.encoder = encoder
        #self.reconst_pixel_log_stdv = tf.get_variable("reconst_pixel_log_stdv", initializer=tf.constant(0.0))

    def get_loss_classification(self, logit, y, class_weights=None, gamma=0.0 ):

        loss = self._ce(logit, y)
        accur = self.get_accuracy(logit, y, gamma)
        return loss, accur

    def get_loss_regression(self, logit, y):
        logit  = tf.reshape( logit, [-1])
        y      = tf.reshape( y,     [-1])
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logit, y))) + eps)
        #loss = tf.reduce_mean(tf.square(tf.subtract(logit, y)))
        #loss = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(logit, y))) + eps)
        return loss

    def get_loss_pxz(self, x_reconst, x_original, pxz):
        if pxz == 'Bernoulli':
            #loss = tf.reduce_mean( tf.reduce_sum(self._binary_crossentropy(x_original, x_reconst),1)) # reconstruction term
            loss = tf.reduce_mean( self._binary_crossentropy(x_original, x_reconst)) # reconstruction term
        elif pxz == 'LeastSquare':
            x_reconst  = tf.reshape( x_reconst, [-1])
            x_original = tf.reshape( x_original, [-1])
            #loss = tf.sqrt(tf.square(tf.reduce_mean(tf.subtract(x_original, x_reconst))) + eps)
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_original, x_reconst))) + eps)
        elif pxz == 'PixelSoftmax':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_reconst, labels=tf.cast(x_original, dtype=tf.int32))) / (self.d.img_size * 256)
        elif pxz == 'DiscretizedLogistic':
            loss = -tf.reduce_mean( self._discretized_logistic(x_reconst, x_original))
        else:
            sys.exit('invalid argument')
        return loss

    def _binary_crossentropy(self, t,o):
        t = tf.reshape( t, (-1, self.d.img_size))
        o = tf.reshape( o, (-1, self.d.img_size))
        return -tf.reduce_sum((t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps)), axis=1)

    def _discretized_logistic(self, x_reconst, x_original, binsize=1/256.0):
        # https://github.com/openai/iaf/blob/master/tf_utils/
        scale = tf.exp(self.reconst_pixel_log_stdv)
        x_original = (tf.floor(x_original / binsize) * binsize - x_reconst) / scale

        logp = tf.log(tf.sigmoid(x_original + binsize / scale) - tf.sigmoid(x_original) + eps)

        shape = x_reconst.get_shape().as_list()
        if len(shape) == 2:   # 1d
            indices = (1,2,3)
        elif len(shape) == 4: # cnn as NHWC
            indices = (1)
        else:
            raise ValueError('shape of x is unexpected')

        return tf.reduce_sum(logp, indices)

    def get_logits_variance(self, z):

        """ z: logits (batch_size, n_mc_sampling, n_class)"""

        def tf_cov(x):                                                                                                 
            x = tf.squeeze(x)    # -> (_n, n_class)                                                                    
            mean_x = tf.reduce_mean(x, axis=0, keepdims=True)                                                          
            mx = tf.matmul(tf.transpose(mean_x), mean_x)
            vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
            cov_xx = vx - mx
            return cov_xx

        def mean_diag_cov(x):
            # after upgrade tf
            #cov = tf.covariance(x)     # -> (_n, _n)
            cov = tf_cov(x)     # -> (_n, _n)
            eigenvalues,_ = tf.linalg.eigh(cov)
            o = tf.reduce_mean( eigenvalues)
            return o

        mean_diag_covs_z = tf.map_fn(mean_diag_cov, z)
        return tf.reduce_mean(mean_diag_covs_z)

    def get_loss_vae(self, dim_z, mu,logsigma, _lambda=1.0 ):

        """  KL( z_L || N(0,I))  """
        #mu, logsigma = mu_logsigma
        sigma = tf.exp(logsigma)
        sigma2 = tf.square(sigma)
       
        kl = 0.5*tf.reduce_sum( (tf.square(mu) + sigma2) - 2*logsigma, 1) - dim_z*.5
        return tf.reduce_mean( tf.maximum(_lambda, kl ))

    def get_loss_kl(self, m, _lambda=1.0 ):

        L = m.L
        Z_SIZES = m.Z_SIZES

        """ KL divergence KL( q(z_l) || p(z_0)) at each lyaer, where p(z_0) is set as N(0,I) """
        Lzs1 = [0]*L

        """ KL( q(z_l) || p(z_l)) to monitor the activities of latent variable units at each layer
             as Fig.4 in http://papers.nips.cc/paper/6275-ladder-variational-autoencoders.pdf """
        Lzs2 = [0]*L

        for l in range(L):
            d_mu, d_logsigma = m.d_mus[l], m.d_logsigmas[l]
            p_mu, p_logsigma = m.p_mus[l], m.p_logsigmas[l]
    
            d_sigma = tf.exp(d_logsigma)
            p_sigma = tf.exp(p_logsigma)
            d_sigma2, p_sigma2 = tf.square(d_sigma), tf.square(p_sigma)
       
            kl1 = 0.5*tf.reduce_sum( (tf.square(d_mu) + d_sigma2) - 2*d_logsigma, 1) - Z_SIZES[l]*.5
            kl2 = 0.5*tf.reduce_sum( (tf.square(d_mu - p_mu) + d_sigma2)/p_sigma2 - 2*tf.log((d_sigma/p_sigma) + eps), 1) - Z_SIZES[l]*.5
    
            Lzs1[l] = tf.reduce_mean( tf.maximum(_lambda, kl1 ))
            Lzs2[l] = tf.reduce_mean( kl2 )
    
        """ use only KL-divergence at the top layer, KL( z_L || z_0) as loss cost for optimaization  """
        loss = Lzs1[-1]
        #loss += tf.add_n(Lzs2)
        return Lzs1, Lzs2, loss

    def get_loss_mmd(self, x, y):
        """
        https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb
        """

        def _kernel(x, y):
            x_size = tf.shape(x)[0]
            y_size = tf.shape(y)[0]
            dim = tf.shape(x)[1]
            tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
            tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
            return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))
    
        x_kernel = _kernel(x, x)
        y_kernel = _kernel(y, y)
        xy_kernel = _kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

    def get_loss_kl_logit(self, logit_real, logit_virtual, mode, training_target_is):
        if training_target_is == 'real':
            logit_virtual = tf.stop_gradient(logit_virtual)
        elif training_target_is == 'virtual':
            logit_real = tf.stop_gradient(logit_real)
        else:
            raise ValueError('unexpected string was set in arg training_target_is.')
        #loss = self._kl_divergence_with_logit(logit_real, logit_virtual)
        if mode == 'kl_forward':
            loss = self._kl_divergence_with_logit(logit_real, logit_virtual)
        elif mode == 'kl_reverse':
            loss = self._kl_divergence_with_logit(logit_virtual, logit_real)
        elif mode == 'js':
            loss = self._js_divergence_with_logit(logit_real, logit_virtual)
        else:
            raise ValueError('unexpected string was set in arg mode.')
        return tf.identity(loss, name="loss_kl_logit")

    def get_loss_pi(self, x, logit_real, is_train):
        logit_real = tf.stop_gradient(logit_real)
        logit_virtual = self.encoder(x, is_train=is_train)
        if c.DIVERGENCE == 'mmd':
            loss = self.get_loss_mmd(logit_virtual, logit_real)
        elif c.DIVERGENCE == 'js':
            loss = self._js_divergence_with_logit(logit_real, logit_virtual)
        elif c.DIVERGENCE == 'least_square':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logit_real, logit_virtual))) + eps)
        else:
            sys.exit('invalid args: %s'%(c.DIVERGENCE))
        return logit_real, logit_virtual, loss

    def get_loss_logit_diff_org(self, logit, y_true):

        # [ToDo] by replacing y_true with y_pred it will turn to be unsupervised way

        real = tf.reduce_sum((y_true)*logit,1)
        other = tf.reduce_max((1-y_true)*logit - (y_true*10000),1)

        IS_TARGETED_ATTACK = False
        if IS_TARGETED_ATTACK:
            # if targetted, optimize for making the other class most likely
            loss = tf.maximum(0.0, other-real+CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss = tf.maximum(0.0, real-other+CONFIDENCE)
        print('loss:', loss)
        return loss

    def get_loss_logit_diff(self, logit_real, logit_virtual, y_true):

        # [ToDo] by replacing y_true with y_pred it will turn to be unsupervised way

        real1 = tf.reduce_mean((y_true)*logit_real,1)
        real2 = tf.reduce_mean((y_true)*logit_virtual,1)

        if c.IS_RELAXED_KL_ENABLE:
            loss = self._kl_divergence_with_logit((y_true)*logit_real, (y_true)*logit_virtual)
        else:
            loss = tf.sqrt(tf.reduce_mean(tf.abs( (y_true)*(logit_virtual - logit_real ) )) + eps)
        """
        #loss = tf.abs(real1 - real2)
        #loss = tf.reduce_mean((y_true)*(logit_virtual - logit_real ),1)
        loss = tf.reduce_mean((y_true)*(tf.abs(logit_virtual - logit_real )),1)
        #real1 = tf.nn.softmax(real1)
        #real2 = tf.nn.softmax(real2)
        #loss = tf.maximum(0.0, real1 - real2)
        #return (loss * 10000), real1, real2
        """
        return loss, real1, real2

    def get_loss_virtual_cw(self, x, logit_real, y_true, is_train):
        r_vadv = self._generate_virtual_cw_perturbation(x, logit_real, y_true, is_train )
        #print(logit_real, r_vadv)
        logit_real = tf.stop_gradient(logit_real)
        logit_virtual = self.encoder(x + r_vadv, is_train=is_train)
        loss, real1, real2 = self.get_loss_logit_diff(logit_real, logit_virtual, y_true)
        return tf.identity(loss, name="vcw_loss"), real1, real2

    def get_loss_vat(self, x, logit_real, is_train, y=None):
        r_vadv = self._generate_virtual_adversarial_perturbation(x, logit_real, is_train, y )
        #print(logit_real, r_vadv)
        logit_real = tf.stop_gradient(logit_real)
        logit_virtual = self.encoder(x + r_vadv, is_train=is_train, y=y)

        if c.DIVERGENCE == 'mmd':
            loss = self.get_loss_mmd(logit_virtual, logit_real)
        elif c.DIVERGENCE == 'kl_forward':
            loss = self._kl_divergence_with_logit(logit_virtual, logit_real)
        elif c.DIVERGENCE == 'kl_reverse':
            loss = self._kl_divergence_with_logit(logit_real, logit_virtual)
        else:
            sys.exit('invalid args: %s'%(c.DIVERGENCE))
        return tf.identity(loss, name="vat_loss"), logit_real, logit_virtual

    def get_loss_fgsm(self, x, y, loss, is_train, is_fgsm=True, name="at_loss"):
        r_adv = self._generate_adversarial_perturbation(x, loss, is_fgsm)
        logit = self.encoder(x + r_adv, is_train=is_train)
        loss = self._ce(logit, y)
        return loss

    def _get_normalized_vector(self, d):

        shape = d.get_shape().as_list()
        if len(shape) == 2:   # 1d
            indices = (1,2,3)
        elif len(shape) == 3: # time-major sequential data as (T, N, embedding dimension)
            indices = (2)
        elif len(shape) == 4: # cnn as NHWC
            indices = (1)
        else:
            raise ValueError('shape of d is unexpected: %s'%(shape))

        d /= (1e-12 + tf.reduce_max(tf.abs(d), indices, keepdims=True))
        d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), indices, keepdims=True))
        return d
    
    def _generate_virtual_cw_perturbation(self, x, logit_real, y_true, is_train ):
        d = tf.random_normal(shape=tf.shape(x))
    
        for _ in range(N_POWER_ITER):
            d = XI * self._get_normalized_vector(d)
            logit_virtual = self.encoder(x + d, is_train=is_train)
            dist, _, _ = self.get_loss_logit_diff( logit_real, logit_virtual, y_true)
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)
    
        return EP * self._get_normalized_vector(d)

    def _generate_virtual_adversarial_perturbation(self, x, logit_real, is_train, y ):
        d = tf.random_normal(shape=tf.shape(x))
    
        for _ in range(N_POWER_ITER):
            d = XI * self._get_normalized_vector(d)
            logit_virtual = self.encoder(x + d, is_train=is_train, y=y)
            dist = self._kl_divergence_with_logit(logit_real, logit_virtual)
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = tf.stop_gradient(grad)
    
        return EP * self._get_normalized_vector(d)

    def _generate_adversarial_perturbation(self, x, loss, is_fgsm):
        grad = tf.gradients(loss, [x], aggregation_method=2)[0]
        grad = tf.stop_gradient(grad)
        norm = self._get_normalized_vector(grad)
        if is_fgsm:
            return c.EPSILON_FGSM * tf.sign(norm)
        else:
            return c.EPSILON_FGSM * norm

    """ https://github.com/takerum/vat_tf/blob/master/layers.py """
    def _ce(self, logit, y):
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)
        #if  self.d.class_weights is None:
        if  not hasattr( self.d, "class_weights"):
            return tf.reduce_mean(unweighted_losses)
        else:
            """ https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy """
            weights = tf.reduce_sum(self.d.class_weights * y, axis=1)
            weighted_losses = unweighted_losses * weights
            return tf.reduce_mean(weighted_losses)

    def get_accuracy(self, logit, y, gamma=0.0):
        pred = tf.argmax(logit, 1)
        true = tf.argmax(y, 1)
        return tf.reduce_mean(tf.to_float(tf.equal(pred, true)))

    def get_accuracy_w_rejection(self, logit, y, gamma=0.0):
    
        """ gamma: confidence threshold """
        prob   = tf.reduce_max( tf.nn.softmax(logit), 1)
        pred   = tf.cast( tf.argmax(logit, 1), tf.int32)
        true   = tf.cast( tf.argmax(y, 1), tf.int32)
        is_hit = tf.cast( tf.equal(pred, true), tf.bool)
        accr   = tf.reduce_mean(tf.to_float(is_hit))
    
        """ accuracy with rejecting unconfident examples """

        """ [ToDo] replace with tf.bitwise.invert after upgrading to TF 1.4 """
        cond      = tf.greater(prob, gamma)
        cond_inv  = tf.less_equal(prob, gamma)
        idxes     = tf.reshape( (tf.where( cond )), [-1])
        idxes_inv = tf.reshape( (tf.where( cond_inv )), [-1])
        n_examples = tf.size(pred)
        n_rejected = n_examples - tf.size(idxes)

        pred_confident = tf.gather( pred, idxes)
        true_confident = tf.gather( true, idxes)
        accr_limited_in_w_confidence = tf.reduce_mean(tf.to_float(tf.equal(pred_confident, true_confident)))
        accr_w_confidence            = tf.reduce_sum(tf.to_float(tf.equal(pred_confident, true_confident))) / tf.to_float(n_examples)

        """ info about error examples """
        cond_error  = tf.not_equal(pred, true)
        idxes_error = tf.reshape( (tf.where( cond_error )), [-1])
        pred_error  = tf.gather( pred, idxes_error)
        true_error  = tf.gather( true, idxes_error)


        o = dict()
        o['accur'] = (accr, accr_limited_in_w_confidence, accr_w_confidence)
        o['n']     = (n_examples, n_rejected)
        o['error'] = (pred_error, true_error)
        o['data']  = (pred, true, prob, is_hit)
        
        return o


    def _logsoftmax(self, x):
        xdev = x - tf.reduce_max(x, 1, keepdims=True)
        lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keepdims=True))
        return lsm

    def _kl_divergence_with_logit(self, q_logit, p_logit):
        q = tf.nn.softmax(q_logit)
        qlogq = tf.reduce_mean(tf.reduce_sum(q * self._logsoftmax(q_logit), 1))
        qlogp = tf.reduce_mean(tf.reduce_sum(q * self._logsoftmax(p_logit), 1))
        return qlogq - qlogp

    def _js_divergence_with_logit(self, q_logit, p_logit):
        q = tf.nn.softmax(q_logit)
        p = tf.nn.softmax(p_logit)
        m = (q + p)/2
        qlogq = tf.reduce_mean(tf.reduce_sum(q * self._logsoftmax(q_logit), 1))
        plogp = tf.reduce_mean(tf.reduce_sum(p * self._logsoftmax(p_logit), 1))
        qlogm = tf.reduce_mean(tf.reduce_sum(q * tf.log(m + eps)))
        plogm = tf.reduce_mean(tf.reduce_sum(p * tf.log(m + eps)))
        return (qlogq + plogp - qlogm - plogm)/2

    def get_loss_entropy_yx(self, logit):
        p = tf.nn.softmax(logit)
        return -tf.reduce_mean(tf.reduce_sum(p * self._logsoftmax(logit), 1))
