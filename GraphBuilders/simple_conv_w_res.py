import tensorflow as tf
from .utils import *

def build_graph(tstep_size=1e-3):
    xdim = 1024+128
    ydim = 3862
    X = tf.placeholder(tf.float32, shape=[None, xdim], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, ydim], name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    
    x_img, x_audio = tf.split(X, [1024, 128], axis=-1)

    img_reshape = tf.reshape(x_img, [-1, 32, 32, 1])

    h1 = lrelu(conv2d(img_reshape, 1, 16, 'conv_1')) # 32x32x1 -> 16x16x16
    h2 = conv2d_with_pooling(h1, 16, 32, 'conv_2')# 16x16x16 -> 8x8x32
    h2_flat = tf.reshape(h2,[-1, 8*8*32])
    h1_flat = tf.reshape(h1,[-1, 16*16*16])

    concat = tf.concat([h1_flat, h2_flat, x_audio], 1)
    h = lrelu(dense(concat, 16*16*16 + 8*8*32 + 128, 100, scope='h'))
    pred = dense(h, 100, ydim, scope = 'pred')

    
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=pred)
    mloss = tf.reduce_mean(loss)

    train = tf.train.AdamOptimizer(tstep_size).minimize(loss)
    
    return X, Y, keep_prob, pred, loss, mloss, train