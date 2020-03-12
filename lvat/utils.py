import tensorflow as tf
import os

def restore(sess, scope, ckpt):
    vars_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    saver = tf.train.Saver(vars_to_restore)
    saver.restore(sess, os.path.join(ckpt,"model.ckpt"))
    return

def is_inited_or_not(sess):
    print('is_inited_or_not() was called')
    for var in tf.global_variables():
        try:
            sess.run(var)
            print('inited:', var.name)
        except tf.errors.FailedPreconditionError:
            print('uninited:', var.name)
    return

def init_uninitialized_vars(sess):

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    
    print('uninitialized_vars to be initialized right now >>>>>>>>>>>>>>>>>>>>>>')
    for var in uninitialized_vars:
        print(var)
    op_init = tf.variables_initializer(uninitialized_vars)
    return op_init 
    
    print("... do init variables in sanitizer")
    self.sess.run(self.op_init)

    return 
