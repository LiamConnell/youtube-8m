import tensorflow as tf
from .utils import *
import tensorflow.contrib.slim as slim


def build_graph(tstep_size=1e-3):
    xdim = 1024+128
    ydim = 3862
    X = tf.placeholder(tf.float32, shape=[None, xdim], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, ydim], name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

#     feature_dim = len(X.get_shape()) - 1
#     X_norm = tf.nn.l2_normalize(X, feature_dim)    
    
    
    x_img, x_audio = tf.split(X, [1024, 128], axis=-1)
    
    a = layers.fully_connected(x_img, 32*32, activation_fn=tf.nn.softmax) # [-1, 32 * 32]

    img_reshape = tf.reshape(x_img, [-1, 32, 32, 1])
    
    h1 =  layers.conv2d(inputs = img_reshape, num_outputs = 32, kernel_size = [7,7]) # [-1, 32, 32, 32]
    
    a_reshape = tf.reshape(tf.tile(tf.reshape(a, [-1, 32,32]), [1,1,32]), [-1,32,32,32])
    
    h1_t = tf.transpose(h1, [0,3,1,2])
    h1_a = tf.multiply(h1_t, a_reshape)
    
    flat = tf.reshape(h1_a, [-1, 32*32*32])
    
    h2 = layers.fully_connected(flat, 1000)
    
    sig = layers.fully_connected(h2, ydim, activation_fn=tf.sigmoid)
    
    
     
    if loss_fn == 'CrossEntropyLoss':
        loss = CrossEntropyLoss(sig, Y)
    if loss_fn == 'HingeLoss':
        loss = HingeLoss(sig, Y)
    if loss_fn == 'SoftmaxLoss':
        loss = SoftmaxLoss(sig, Y)
    
    mloss = loss

    train = tf.train.AdamOptimizer(tstep_size).minimize(loss)
    
    return X, Y, keep_prob, sig, loss, mloss, train